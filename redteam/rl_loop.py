"""System 2+3 — self-hardening red-team loop.

The loop that makes DUALMIND get *stronger as it is attacked*:

  1. Measure the current Attack Success Rate (ASR) of an attack pool against the
     defender (the retrainable pre-filter classifier — the fast path).
  2. For each attack that currently bypasses, generate semantic-preserving
     mutations (System 2) and keep only genuine paraphrases (System 3).
  3. Fire the surviving mutations at the defender; reward = how close each got to
     bypassing (1 - block_probability).
  4. Confirmed bypasses are added to the positive training set and the classifier
     is retrained. Next round, those variants — and the originals — are caught.

Returns the ASR-over-rounds curve, which should decrease monotonically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from prefilter.embedding_classifier import EmbeddingClassifier, bundled_training_data

from .mutation_engine import MutationEngine
from .semantic_filter import SemanticFilter


@dataclass
class HardeningHistory:
    asr_per_round: List[float] = field(default_factory=list)   # [before, after r1, ...]
    bypasses_per_round: List[int] = field(default_factory=list)
    mutations_fired_per_round: List[int] = field(default_factory=list)
    mean_reward_per_round: List[float] = field(default_factory=list)
    rounds: int = 0

    @property
    def final_asr(self) -> float:
        return self.asr_per_round[-1] if self.asr_per_round else 0.0

    @property
    def initial_asr(self) -> float:
        return self.asr_per_round[0] if self.asr_per_round else 0.0


class SelfHardeningLoop:
    def __init__(
        self,
        classifier: Optional[EmbeddingClassifier] = None,
        mutation_engine: Optional[MutationEngine] = None,
        semantic_filter: Optional[SemanticFilter] = None,
        block_threshold: float = 0.9,
        init_pos: Optional[Sequence[str]] = None,
        init_neg: Optional[Sequence[str]] = None,
    ):
        self.classifier = classifier or EmbeddingClassifier(block_threshold=block_threshold)
        self.mutation_engine = mutation_engine or MutationEngine(provider=None)
        self.semantic_filter = semantic_filter or SemanticFilter()
        self.block_threshold = block_threshold

        if init_pos is None or init_neg is None:
            texts, labels = bundled_training_data()
            self._pos = [t for t, y in zip(texts, labels) if y == 1]
            self._neg = [t for t, y in zip(texts, labels) if y == 0]
        else:
            self._pos = list(init_pos)
            self._neg = list(init_neg)
        self._refit()

    # ------------------------------------------------------------------ #
    def _refit(self) -> None:
        texts = self._pos + self._neg
        labels = [1] * len(self._pos) + [0] * len(self._neg)
        self.classifier.fit(texts, labels)

    def _bypasses(self, attack: str) -> bool:
        """True if the defender FAILS to block (attack succeeds)."""
        return self.classifier.predict_proba(attack) < self.block_threshold

    def _asr(self, attack_pool: Sequence[str]) -> float:
        if not attack_pool:
            return 0.0
        return sum(self._bypasses(a) for a in attack_pool) / len(attack_pool)

    # ------------------------------------------------------------------ #
    def run(
        self,
        attack_pool: Sequence[str],
        rounds: int = 5,
        max_mutations: int = 10,
    ) -> HardeningHistory:
        hist = HardeningHistory(rounds=rounds)
        hist.asr_per_round.append(round(self._asr(attack_pool), 4))

        for _ in range(rounds):
            confirmed: List[str] = []
            fired = 0
            rewards: List[float] = []

            for attack in attack_pool:
                if not self._bypasses(attack):
                    continue  # already caught — focus effort on what still works
                batch = self.mutation_engine.mutate(attack, n=max_mutations)
                kept = self.semantic_filter.keep(attack, batch.mutations)
                for mut in kept:
                    fired += 1
                    prob = self.classifier.predict_proba(mut)
                    rewards.append(1.0 - prob)  # closeness to bypass
                    if prob < self.block_threshold:
                        confirmed.append(mut)
                # the bypassing original is itself a confirmed miss to learn from
                confirmed.append(attack)

            # learn from confirmed bypasses, then retrain
            new_pos = [c for c in confirmed if c not in self._pos]
            self._pos.extend(new_pos)
            if new_pos:
                self._refit()

            hist.bypasses_per_round.append(len(set(confirmed)))
            hist.mutations_fired_per_round.append(fired)
            hist.mean_reward_per_round.append(
                round(sum(rewards) / len(rewards), 4) if rewards else 0.0
            )
            hist.asr_per_round.append(round(self._asr(attack_pool), 4))

        return hist
