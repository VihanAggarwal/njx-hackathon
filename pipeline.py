"""DUALMIND orchestrator — wires all 8 systems into one decision path.

This is the glue the spec's "first trace", the live demo, and the eval harness all
run through. `DualMind.process(...)` takes one (untrusted content, user goal) pair
and returns a full DecisionTrace: which system caught it, the taint labels at each
step, the component + overall risk, the review-gate routing, and the KB entry id.

The component flags let the eval harness build the ablation matrix:
    no defense / pre-filter only / dual-LLM only / dual-LLM+taint / full DUALMIND
by toggling `use_prefilter`, `use_dual_llm`, `use_taint` (and swapping in a
pre vs post self-hardening pre-filter classifier).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from audit import HashChain
from dual_llm import Decider, Reader
from federation import FederatedInstance
from knowledge_base import KBStore
from prefilter import PreFilter
from review_gate import ReviewQueue
from taint import TaintChecker, TaintTracker


@dataclass
class DecisionTrace:
    content: str
    content_type: str
    agent: str
    user_goal: str
    ground_truth: Optional[str] = None

    prefilter: Any = None
    reader: Any = None
    decider: Any = None

    prefilter_risk: float = 0.0
    dual_llm_risk: float = 0.0
    taint_risk: float = 0.0
    risk: float = 0.0

    caught_by: str = "none"
    fast_path: bool = False
    routing: str = "auto_allow"
    final_verdict: str = "allow"          # "allow" | "block" | "review"
    taint_findings: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)

    review_item: Any = None
    kb_id: Optional[int] = None
    audit_head: str = ""
    latency_ms: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.final_verdict == "block"


@dataclass
class DualMindConfig:
    use_prefilter: bool = True
    use_dual_llm: bool = True
    use_taint: bool = True
    use_review_gate: bool = True   # human-in-the-loop oracle (full configs only)
    name: str = "full"


class DualMind:
    def __init__(
        self,
        config: dict,
        provider=None,
        prefilter: Optional[PreFilter] = None,
        flags: Optional[DualMindConfig] = None,
        kb: Optional[KBStore] = None,
        federation: Optional[FederatedInstance] = None,
    ):
        self.config = config
        self.flags = flags or DualMindConfig()
        self.provider = provider
        thr = config.get("thresholds", {})
        self.block_threshold = thr.get("review_gate_block", 0.8)
        self.review_threshold = thr.get("review_gate_review", 0.3)

        self.prefilter = prefilter
        if self.flags.use_prefilter and self.prefilter is None:
            self.prefilter = PreFilter.with_default_classifier(config)

        models = config.get("models", {})
        self.taint = TaintTracker()
        self.checker = TaintChecker()
        self.reader = None
        self.decider = None
        if self.flags.use_dual_llm and provider is not None:
            self.reader = Reader(provider, models.get("reader", "claude-haiku-4-5"), self.taint)
            self.decider = Decider(
                provider, models.get("decider", "claude-opus-4-8"),
                self.taint, self.checker,
            )

        self.review_queue = ReviewQueue(config)
        self.kb = kb if kb is not None else KBStore(":memory:")
        self.federation = federation
        self.audit = HashChain()

    # ------------------------------------------------------------------ #
    def process(
        self,
        content: str,
        user_goal: str = "Summarize this content for me.",
        content_type: str = "text",
        agent: str = "agent-0",
        ground_truth: Optional[str] = None,
    ) -> DecisionTrace:
        t0 = time.perf_counter()
        tr = DecisionTrace(
            content=content, content_type=content_type, agent=agent,
            user_goal=user_goal, ground_truth=ground_truth,
        )
        self.audit.append("ingest", {"agent": agent, "content_type": content_type,
                                     "len": len(content)})

        # --- Stage 0: pre-filter (fast path) ---------------------------- #
        if self.flags.use_prefilter and self.prefilter is not None:
            pf = self.prefilter.score(content, content_type)
            tr.prefilter = pf
            tr.prefilter_risk = pf.risk
            tr.signals.extend(pf.signals)
            self.audit.append("prefilter", {"verdict": pf.verdict, "risk": pf.risk,
                                            "signals": pf.signals})
            if pf.verdict == "block":
                tr.caught_by = "prefilter"
                tr.fast_path = True
                return self._finalize(tr, t0, fast_blocked=True)

        # --- Stage 1+4: dual-LLM (slow path) + taint -------------------- #
        if self.flags.use_dual_llm and self.reader is not None:
            reader_out = self.reader.read(content, content_type)
            tr.reader = reader_out
            self.audit.append("reader", {"suspicious": reader_out.suspicious,
                                         "contains_instructions": reader_out.contains_instructions,
                                         "n_actions": len(reader_out.requested_actions)})
            # dual-LLM (privilege separation) signal — the model's own judgment
            tr.dual_llm_risk = (
                0.85 if reader_out.suspicious
                else 0.5 if reader_out.contains_instructions else 0.05
            )

            decider_out = self.decider.decide(reader_out, user_goal)
            tr.decider = decider_out
            self.audit.append("decider", {"decision": decider_out.decision,
                                          "n_calls": len(decider_out.calls),
                                          "risk": decider_out.risk})
            if self.flags.use_taint:
                tr.taint_risk = decider_out.risk
                for c in decider_out.flagged_calls:
                    tr.taint_findings.append(f"{c.tool}: {c.taint.reason}")
                self.audit.append("taint_check",
                                  {"flagged": bool(decider_out.flagged_calls),
                                   "findings": tr.taint_findings})

        # --- overall risk over enabled components ----------------------- #
        components = []
        if self.flags.use_prefilter:
            components.append(tr.prefilter_risk)
        if self.flags.use_dual_llm:
            components.append(tr.dual_llm_risk)
        if self.flags.use_taint:
            components.append(tr.taint_risk)
        tr.risk = round(max(components), 4) if components else 0.0

        # --- Stage 6: review gate --------------------------------------- #
        item = self.review_queue.submit(
            content=content, agent=agent, risk_score=tr.risk,
            signals=tr.signals, taint_trace=tr.taint_findings,
            ground_truth=ground_truth,
        )
        tr.review_item = item
        tr.routing = item.routing
        self.audit.append("review_gate", {"routing": item.routing, "risk": tr.risk})
        if (self.flags.use_review_gate and item.routing == "review"
                and ground_truth is not None):
            self.review_queue.simulate_decision(item)

        return self._finalize(tr, t0)

    # ------------------------------------------------------------------ #
    def _attribute(self, tr: DecisionTrace) -> str:
        """Which enabled system is responsible for a block.

        A block decided in the human-review band is the review gate's catch, not a
        detector's — credit it to review_gate, not whichever detector happened to
        have the highest (sub-threshold) risk.
        """
        if self.flags.use_review_gate and tr.routing == "review":
            return "review_gate"
        if self.flags.use_taint and tr.taint_risk >= self.block_threshold:
            return "taint"
        if self.flags.use_dual_llm and tr.dual_llm_risk >= self.block_threshold:
            return "dual_llm"
        if self.flags.use_prefilter and tr.prefilter_risk >= self.block_threshold:
            return "prefilter"
        return "review_gate"

    def _finalize(self, tr: DecisionTrace, t0: float, fast_blocked: bool = False) -> DecisionTrace:
        if fast_blocked:
            tr.final_verdict = "block"
            tr.routing = "auto_block"
        elif self.flags.use_review_gate and tr.review_item is not None:
            # Full system: the human-in-the-loop oracle decides the review band.
            tr.final_verdict = self.review_queue.final_verdict(tr.review_item)
            if tr.final_verdict == "block" and tr.caught_by == "none":
                tr.caught_by = self._attribute(tr)
        else:
            # Detection-only configs: automated block at the block threshold (no
            # human review band).
            tr.final_verdict = "block" if tr.risk >= self.block_threshold else "allow"
            if tr.final_verdict == "block" and tr.caught_by == "none":
                tr.caught_by = self._attribute(tr)

        # knowledge base entry
        tr.kb_id = self.kb.add({
            "content": tr.content, "content_type": tr.content_type, "agent": tr.agent,
            "risk": tr.risk if not fast_blocked else tr.prefilter_risk,
            "signals": tr.signals, "taint_trace": tr.taint_findings,
            "routing": tr.routing, "caught_by": tr.caught_by,
            "fast_path": tr.fast_path,
            "human_decision": (tr.review_item.human_decision if tr.review_item else None),
        })

        # federation: share a novel SLOW-path bypass so peers catch it early
        if (self.federation is not None and tr.blocked and not tr.fast_path):
            self.federation.learn_local(tr.content)

        tr.audit_head = self.audit.head
        tr.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return tr

    # ------------------------------------------------------------------ #
    def render_trace(self, tr: DecisionTrace) -> str:
        lines = [
            "=" * 70,
            "DUALMIND DECISION TRACE",
            "=" * 70,
            f"agent        : {tr.agent}",
            f"content_type : {tr.content_type}",
            f"user goal    : {tr.user_goal}",
            f"content      : {tr.content[:160]}{'...' if len(tr.content) > 160 else ''}",
            "-" * 70,
        ]
        if tr.prefilter is not None:
            ss = tr.prefilter.stage_scores
            lines.append(f"[0] PRE-FILTER  verdict={tr.prefilter.verdict} "
                         f"risk={tr.prefilter_risk:.3f} latency={tr.prefilter.latency_ms:.2f}ms")
            lines.append(f"      stages: pattern={ss['pattern']} "
                         f"classifier={ss['classifier']} structural={ss['structural']}")
            lines.append(f"      signals: {', '.join(tr.signals[:8]) or '(none)'}")
        if tr.reader is not None:
            lines.append(f"[1] READER      suspicious={tr.reader.suspicious} "
                         f"contains_instructions={tr.reader.contains_instructions} "
                         f"actions={len(tr.reader.requested_actions)}")
            labels = ", ".join(f"{k}={v.label.value}"
                               for k, v in list(tr.reader.tainted.items())[:5])
            lines.append(f"      taint labels: {labels or '(none)'}")
        if tr.decider is not None:
            lines.append(f"[1] DECIDER     decision={tr.decider.decision} "
                         f"tool_calls={len(tr.decider.calls)} risk={tr.decider.risk:.3f}")
            for c in tr.decider.calls:
                flag = "  <-- TAINT FLAGGED" if c.flagged else ""
                argstr = ", ".join(f"{k}={getattr(v, 'value', v)}" for k, v in c.args.items())
                lines.append(f"        call {c.tool}({argstr}){flag}")
        if tr.taint_findings:
            lines.append(f"[4] TAINT       {'; '.join(tr.taint_findings)}")
        lines += [
            "-" * 70,
            f"[6] REVIEW GATE routing={tr.routing}",
            f"OVERALL RISK  : {tr.risk:.3f}   "
            f"(prefilter={tr.prefilter_risk:.2f} dual_llm={tr.dual_llm_risk:.2f} "
            f"taint={tr.taint_risk:.2f})",
            f"CAUGHT BY     : {tr.caught_by}   fast_path={tr.fast_path}",
            f"FINAL VERDICT : {tr.final_verdict.upper()}",
            f"[7] KB entry  : #{tr.kb_id}",
            f"AUDIT head    : {tr.audit_head[:24]}...",
            f"latency       : {tr.latency_ms:.2f} ms",
            "=" * 70,
        ]
        return "\n".join(lines)
