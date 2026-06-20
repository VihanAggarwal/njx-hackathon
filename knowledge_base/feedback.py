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

        Human rejections are placed first (highest priority).
        """
        ts = TrainingSet()
        rows = self.kb.all()
        # highest priority: human-confirmed attacks
        for r in rows:
            if r.get("human_decision") == REJECT:
                ts.texts.append(r["content"]); ts.labels.append(1)
        for r in rows:
            if r.get("human_decision") != REJECT and r.get("routing") == "auto_block":
                ts.texts.append(r["content"]); ts.labels.append(1)
        for r in rows:
            if r.get("human_decision") == APPROVE or r.get("routing") == "auto_allow":
                ts.texts.append(r["content"]); ts.labels.append(0)
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
