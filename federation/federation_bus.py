"""System 5 — simulated federated signature-sharing bus.

N DUALMIND instances share abstract bypass signatures over a bus. When instance A
discovers a novel attack, it publishes the (DP-noised) signature; every other
instance ingests it and can now block that attack *before being attacked itself*.

The demo (and tests) show a novel attack at instance A propagating to instance B
ahead of B ever seeing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .signature_extractor import Signature, SignatureExtractor


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


@dataclass
class MatchResult:
    blocked: bool
    similarity: float
    matched_family: bool


class FederatedInstance:
    def __init__(self, name: str, extractor: SignatureExtractor, match_threshold: float = 0.9):
        self.name = name
        self.extractor = extractor
        self.match_threshold = match_threshold
        self.known: List[Signature] = []
        self.bus: Optional["FederationBus"] = None

    def learn_local(self, content: str) -> Signature:
        """Locally discover a bypass, store its signature, and broadcast it."""
        sig = self.extractor.extract(content)
        self.known.append(sig)
        if self.bus is not None:
            self.bus.publish(sig, origin=self.name)
        return sig

    def ingest(self, sig: Signature) -> None:
        self.known.append(sig)

    def check(self, content: str) -> MatchResult:
        """Would this instance block `content` given its known signatures?"""
        sig = self.extractor.extract(content)
        v = sig.to_vector()
        best = 0.0
        family = False
        for known in self.known:
            sim = _cosine(v, known.to_vector())
            if sim > best:
                best = sim
            if known.family_hash == sig.family_hash:
                family = True
        return MatchResult(
            blocked=best >= self.match_threshold, similarity=round(best, 4),
            matched_family=family,
        )


class FederationBus:
    def __init__(self):
        self.instances: Dict[str, FederatedInstance] = {}
        self.published: List[tuple] = []  # (origin, Signature)

    def register(self, instance: FederatedInstance) -> None:
        instance.bus = self
        self.instances[instance.name] = instance

    def publish(self, sig: Signature, origin: str) -> int:
        """Broadcast a signature to every OTHER instance. Returns #recipients."""
        self.published.append((origin, sig))
        recipients = 0
        for name, inst in self.instances.items():
            if name != origin:
                inst.ingest(sig)
                recipients += 1
        return recipients
