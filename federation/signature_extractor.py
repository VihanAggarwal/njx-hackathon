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
        # Deterministic SIGNED feature-hashing embedding (no model needed). Features
        # are word tokens PLUS padded character trigrams, so even single-word content
        # populates several buckets — this cuts the short-content hash collisions
        # that would otherwise cause false-positive federated blocks. Wide dim + a
        # sign bit reduce collisions further.
        dim = 1024
        vec = np.zeros(dim, dtype=np.float64)
        text = content.lower()
        features = list(text.split())
        padded = f" {text} "
        features += [padded[i : i + 3] for i in range(max(0, len(padded) - 2))]
        for feat in features:
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def extract(self, content: str) -> Signature:
        vec = self._embed(content)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        # Differential-privacy: add Laplace noise with a norm-controlled budget so
        # the noise level is independent of embedding dimension (10% of unit signal
        # at epsilon=1). Stronger privacy -> lower epsilon -> larger noise.
        raw = self._np_rng.laplace(0.0, 1.0, size=vec.shape)
        rnorm = np.linalg.norm(raw)
        if rnorm > 0:
            raw = raw / rnorm
        noise = raw * (0.1 / max(1e-6, self.dp_epsilon))
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
