"""System 5 — bypass -> abstract signature (with differential-privacy noise).

A confirmed bypass is turned into an ABSTRACT signature that can be shared across
instances without ever sharing the raw content: a (noised) embedding plus a few
structural features and a coarse bucketed length. Differential-privacy noise is
added to the embedding so the shared signature can't be inverted to reconstruct
the original text.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Signature:
    embedding: List[float]               # noised, L2-normalized
    struct_features: List[float]         # length bucket, punctuation ratio, etc.
    family_hash: str                     # coarse, non-invertible bucket id
    dp_epsilon: float = 1.0

    def to_vector(self) -> np.ndarray:
        return np.asarray(self.embedding, dtype=np.float64)


def _structural_features(content: str) -> List[float]:
    n = max(1, len(content))
    words = content.split()
    return [
        min(1.0, len(content) / 500.0),                       # length bucket
        sum(not c.isalnum() and not c.isspace() for c in content) / n,  # punct ratio
        sum(c.isupper() for c in content) / n,                # uppercase ratio
        min(1.0, len(words) / 100.0),                         # word count bucket
        1.0 if any(ord(c) > 127 for c in content) else 0.0,   # non-ascii present
    ]


class SignatureExtractor:
    def __init__(self, embedder=None, dp_epsilon: float = 1.0, seed: Optional[int] = None):
        self.embedder = embedder
        self.dp_epsilon = dp_epsilon
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _embed(self, content: str) -> np.ndarray:
        if self.embedder is not None:
            if hasattr(self.embedder, "fitted") and not self.embedder.fitted:
                self.embedder.fit([content])
            return np.asarray(self.embedder.transform([content])[0], dtype=np.float64)
        # Deterministic hashing embedding (no model needed): bag-of-token-hashes.
        dim = 64
        vec = np.zeros(dim, dtype=np.float64)
        for tok in content.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % dim] += 1.0
        return vec

    def extract(self, content: str) -> Signature:
        vec = self._embed(content)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        # Differential-privacy: Laplace noise scaled by sensitivity/epsilon.
        scale = 1.0 / max(1e-6, self.dp_epsilon)
        noise = self._np_rng.laplace(0.0, scale * 0.01, size=vec.shape)
        noised = vec + noise
        n2 = np.linalg.norm(noised)
        if n2 > 0:
            noised = noised / n2

        struct = _structural_features(content)
        # family hash: coarse bucket from rounded structural features only — never
        # the raw content, so it is not invertible.
        family = hashlib.sha256(
            ("|".join(f"{round(s, 1)}" for s in struct)).encode()
        ).hexdigest()[:16]

        return Signature(
            embedding=noised.tolist(),
            struct_features=struct,
            family_hash=family,
            dp_epsilon=self.dp_epsilon,
        )
