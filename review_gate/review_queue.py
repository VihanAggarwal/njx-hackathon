"""System 6 — human review gate.

Risk-score routing:
    risk < review_gate_review (0.3)  -> auto_allow
    review <= risk <= block (0.8)    -> human review queue
    risk > review_gate_block (0.8)   -> auto_block

The queue holds each intercepted message with full context (content, agent, taint
trace, risk score, fired signals). In eval mode, "human" decisions are simulated
from the ground-truth label; for the live demo a CLI/web reviewer decides. Human
rejections are surfaced as the highest-priority training signal for System 2.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

AUTO_ALLOW = "auto_allow"
REVIEW = "review"
AUTO_BLOCK = "auto_block"

APPROVE = "approve"
REJECT = "reject"

_ids = itertools.count(1)


@dataclass
class ReviewItem:
    content: str
    agent: str
    risk_score: float
    signals: List[str] = field(default_factory=list)
    taint_trace: List[str] = field(default_factory=list)
    routing: str = REVIEW
    ground_truth: Optional[str] = None      # "attack" | "benign" (eval mode)
    human_decision: Optional[str] = None    # APPROVE | REJECT
    note: str = ""
    timestamp: Optional[str] = None
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def decided(self) -> bool:
        return self.human_decision is not None


class ReviewQueue:
    def __init__(self, config: Optional[dict] = None):
        thr = (config or {}).get("thresholds", {})
        self.block_threshold = thr.get("review_gate_block", 0.8)
        self.review_threshold = thr.get("review_gate_review", 0.3)
        self._items: Dict[int, ReviewItem] = {}
        self._pending: List[int] = []
        # callback invoked with a ReviewItem whenever a human REJECTS (top-priority
        # feedback for the red-team / prefilter retraining).
        self.on_reject: Optional[Callable[[ReviewItem], None]] = None

    # ------------------------------------------------------------------ #
    def route(self, risk: float) -> str:
        if risk > self.block_threshold:
            return AUTO_BLOCK
        if risk >= self.review_threshold:
            return REVIEW
        return AUTO_ALLOW

    def submit(
        self,
        content: str,
        agent: str,
        risk_score: float,
        signals: Optional[List[str]] = None,
        taint_trace: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> ReviewItem:
        item = ReviewItem(
            content=content, agent=agent, risk_score=risk_score,
            signals=signals or [], taint_trace=taint_trace or [],
            routing=self.route(risk_score), ground_truth=ground_truth,
            timestamp=timestamp,
        )
        self._items[item.id] = item
        if item.routing == REVIEW:
            self._pending.append(item.id)
        return item

    # ------------------------------------------------------------------ #
    def pending(self) -> List[ReviewItem]:
        return [self._items[i] for i in self._pending if not self._items[i].decided]

    def get(self, item_id: int) -> Optional[ReviewItem]:
        return self._items.get(item_id)

    def decide(self, item_id: int, decision: str, note: str = "") -> ReviewItem:
        item = self._items[item_id]
        item.human_decision = decision
        item.note = note
        if item_id in self._pending:
            self._pending.remove(item_id)
        if decision == REJECT and self.on_reject is not None:
            self.on_reject(item)
        return item

    def simulate_decision(self, item: ReviewItem) -> ReviewItem:
        """Eval-mode 'human': reject attacks, approve benign, from ground truth."""
        if item.ground_truth is None:
            raise ValueError("simulate_decision requires a ground_truth label")
        decision = REJECT if item.ground_truth == "attack" else APPROVE
        return self.decide(item.id, decision, note="simulated from ground_truth")

    def drain_simulated(self) -> List[ReviewItem]:
        decided = []
        for item in list(self.pending()):
            decided.append(self.simulate_decision(item))
        return decided

    # ------------------------------------------------------------------ #
    def final_verdict(self, item: ReviewItem) -> str:
        """Collapse routing + human decision into block/allow."""
        if item.routing == AUTO_BLOCK:
            return "block"
        if item.routing == AUTO_ALLOW:
            return "allow"
        # human-reviewed
        if item.human_decision == REJECT:
            return "block"
        if item.human_decision == APPROVE:
            return "allow"
        return "pending"

    def rejections(self) -> List[ReviewItem]:
        return [i for i in self._items.values() if i.human_decision == REJECT]

    def approvals(self) -> List[ReviewItem]:
        return [i for i in self._items.values() if i.human_decision == APPROVE]
