"""System 7 — three feedback paths.

The knowledge base closes the loop by turning intercept history into training
signal along three documented paths:

  1. rejections  -> pre-filter retraining set (confirmed attacks become positives)
  2. approvals   -> system-prompt allow-list (confirmed-benign patterns)
  3. novel high-risk / rejected patterns -> mutation corpus for System 2

Human rejections are the highest-priority signal (they are ground-truth attacks a
human confirmed), so they are weighted first in the training set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .kb_store import KBStore

REJECT = "reject"
APPROVE = "approve"


@dataclass
class TrainingSet:
    texts: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.texts)


class FeedbackRouter:
    def __init__(self, kb: KBStore):
        self.kb = kb

    # path 1 -------------------------------------------------------------- #
    def prefilter_training_set(self) -> TrainingSet:
        """Attacks (rejected or auto-blocked) -> 1; benign (approved/allowed) -> 0.

        Each row contributes EXACTLY ONE label, by precedence — a human decision
        always wins over the automated routing (so a reviewer's correction of a
        false positive/negative is honored, never duplicated or contradicted).
        Human rejections are emitted first (highest-priority signal).
        """
        rejects: List[str] = []
        auto_blocks: List[str] = []
        benign: List[str] = []
        for r in self.kb.all():
            hd = r.get("human_decision")
            routing = r.get("routing")
            if hd == REJECT:                 # human-confirmed attack (wins)
                rejects.append(r["content"])
            elif hd == APPROVE:              # human-confirmed benign (wins)
                benign.append(r["content"])
            elif routing == "auto_block":
                auto_blocks.append(r["content"])
            elif routing == "auto_allow":
                benign.append(r["content"])

        ts = TrainingSet()
        for c in rejects:
            ts.texts.append(c); ts.labels.append(1)
        for c in auto_blocks:
            ts.texts.append(c); ts.labels.append(1)
        for c in benign:
            ts.texts.append(c); ts.labels.append(0)
        return ts

    # path 2 -------------------------------------------------------------- #
    def allowlist(self) -> List[str]:
        """Confirmed-benign content patterns for the system-prompt allow-list."""
        out = []
        for r in self.kb.all():
            if r.get("human_decision") == APPROVE:
                out.append(r["content"])
        return out

    # path 3 -------------------------------------------------------------- #
    def mutation_corpus(self, risk_threshold: float = 0.6) -> List[str]:
        """Novel attack patterns to feed the red-team mutation engine (System 2).

        Anything a human rejected, plus high-risk near-misses that weren't
        auto-blocked (the interesting, evolving frontier).
        """
        seen = set()
        out: List[str] = []
        for r in self.kb.all():
            is_novel = (
                r.get("human_decision") == REJECT
                or (r.get("risk", 0.0) >= risk_threshold and r.get("routing") != "auto_block")
            )
            if is_novel and r["content"] not in seen:
                seen.add(r["content"])
                out.append(r["content"])
        return out
