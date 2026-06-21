"""System 0 — Pre-filter.

Three sequential stages, designed to run in <50ms on CPU:
  1. PatternIndex        — known-signature / homoglyph / zero-width / base64 scan
  2. EmbeddingClassifier — calibrated injection probability
  3. StructuralAnomaly   — content-vs-claimed-structure mismatch

`PreFilter.score(content, content_type)` runs all three and returns a
`PrefilterResult` carrying per-stage scores plus an overall verdict.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from .embedding_classifier import (
    ClassifierResult,
    EmbeddingClassifier,
    bundled_training_data,
)
from .pattern_index import PatternIndex, PatternResult
from .structural_anomaly import AnomalyResult, StructuralAnomaly

__all__ = [
    "PreFilter",
    "PrefilterResult",
    "PatternIndex",
    "PatternResult",
    "EmbeddingClassifier",
    "ClassifierResult",
    "StructuralAnomaly",
    "AnomalyResult",
]


def _combine(*scores: float) -> float:
    prod = 1.0
    for s in scores:
        prod *= (1.0 - max(0.0, min(1.0, s)))
    return round(1.0 - prod, 4)


@dataclass
class PrefilterResult:
    verdict: str                       # "block" | "near_miss" | "pass"
    risk: float                        # overall combined risk 0..1
    pattern: PatternResult = None
    structural: AnomalyResult = None
    classifier: Optional[ClassifierResult] = None
    classifier_prob: Optional[float] = None   # raw calibrated prob (for ECE)
    latency_ms: float = 0.0
    signals: list = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.verdict == "block"

    @property
    def stage_scores(self) -> dict:
        return {
            "pattern": self.pattern.score if self.pattern else None,
            "classifier": self.classifier_prob,
            "structural": self.structural.score if self.structural else None,
        }


class PreFilter:
    def __init__(
        self,
        config: Optional[dict] = None,
        classifier: Optional[EmbeddingClassifier] = None,
    ):
        thr = (config or {}).get("thresholds", {})
        self.block_threshold = thr.get("prefilter_block", 0.9)
        self.near_miss_threshold = thr.get("prefilter_near_miss", 0.6)
        self.pattern_index = PatternIndex()
        self.structural = StructuralAnomaly()
        self.classifier = classifier
        # In-memory LRU cache: repeated/near-repeated content returns instantly.
        self._cache_enabled = (config or {}).get("perf", {}).get("prefilter_cache", True)
        self._cache: "OrderedDict[str, PrefilterResult]" = OrderedDict()
        self._cache_max = 8192
        self.cache_hits = 0

    # ------------------------------------------------------------------ #
    @classmethod
    def with_default_classifier(cls, config: Optional[dict] = None) -> "PreFilter":
        """Build a PreFilter with a classifier trained on the bundled seed set.

        For real benchmarks the eval harness trains on LLMail-Inject and injects
        that classifier instead; this is the offline / demo convenience path.
        """
        thr = (config or {}).get("thresholds", {})
        clf = EmbeddingClassifier(
            block_threshold=thr.get("prefilter_block", 0.9),
            near_miss_threshold=thr.get("prefilter_near_miss", 0.6),
        )
        texts, labels = bundled_training_data()
        clf.fit(texts, labels)
        return cls(config=config, classifier=clf)

    def set_classifier(self, classifier: EmbeddingClassifier) -> None:
        self.classifier = classifier

    # ------------------------------------------------------------------ #
    def score(self, content: str, content_type: str = "text") -> PrefilterResult:
        t0 = time.perf_counter()

        key = None
        if self._cache_enabled:
            key = hashlib.sha256((content_type + "\x00" + content).encode("utf-8")).hexdigest()
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return cached

        pattern = self.pattern_index.score(content, content_type)
        structural = self.structural.score(content, content_type)

        clf_result: Optional[ClassifierResult] = None
        clf_prob: Optional[float] = None
        if self.classifier is not None and self.classifier._clf is not None:
            clf_result = self.classifier.classify(content)
            clf_prob = clf_result.prob

        # Overall risk: pattern hits and the calibrated classifier prob are the
        # primary drivers; structural anomaly is a secondary amplifier.
        risk = _combine(pattern.score, clf_prob or 0.0, 0.6 * structural.score)

        # A high-confidence signature alone forces a block (fast path).
        if pattern.score >= self.block_threshold or risk >= self.block_threshold:
            verdict = "block"
        elif risk >= self.near_miss_threshold:
            verdict = "near_miss"
        else:
            verdict = "pass"

        signals = list(pattern.signals)
        signals += [f"struct:{r}" for r in structural.reasons]
        if clf_result is not None:
            signals.append(f"clf:{clf_result.verdict}={clf_prob:.2f}")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = PrefilterResult(
            verdict=verdict,
            risk=risk,
            pattern=pattern,
            structural=structural,
            classifier=clf_result,
            classifier_prob=clf_prob,
            latency_ms=latency_ms,
            signals=signals,
        )
        if key is not None:
            self._cache[key] = result
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return result
