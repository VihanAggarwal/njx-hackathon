"""System 3 — semantic invariance filter.

Embeds the original attack and each candidate mutation, then rejects any mutation
whose cosine similarity to the original falls below a threshold (default 0.85).
Only genuine paraphrases — same meaning, different surface — survive to be fired
at the defender. This stops the mutation engine from "winning" by drifting to a
different (benign) sentence.

Uses sentence-transformers when available; otherwise a TF-IDF embedding fit on
the small (original + mutations) corpus. The 0.85 threshold is calibrated for
sentence embeddings; the TF-IDF fallback uses a lower default — documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class FilteredMutation:
    text: str
    similarity: float
    kept: bool


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticFilter:
    def __init__(self, threshold: float = 0.85, embedder: Optional[object] = None):
        self.embedder = embedder
        self._backend = None
        if embedder is not None:
            self._backend = getattr(embedder, "backend", "custom")
            self.threshold = threshold
        else:
            self._backend, self.threshold = self._auto_threshold(threshold)

    @staticmethod
    def _auto_threshold(requested: float):
        """Pick an embedder and a threshold appropriate for it."""
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            return "sentence-transformers", requested
        except Exception:
            # TF-IDF paraphrase similarities run lower; relax unless caller overrode.
            return "tfidf", min(requested, 0.55)

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.embedder is not None:
            if hasattr(self.embedder, "fitted") and not self.embedder.fitted:
                self.embedder.fit(texts)
            return self.embedder.transform(texts)
        if self._backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return np.asarray(model.encode(texts, normalize_embeddings=True))
        # TF-IDF fit on this small corpus
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
        return vec.fit_transform(texts).toarray()

    def filter(self, original: str, mutations: List[str]) -> List[FilteredMutation]:
        if not mutations:
            return []
        embs = self._embed([original] + list(mutations))
        base = embs[0]
        results: List[FilteredMutation] = []
        for i, mut in enumerate(mutations, start=1):
            sim = _cosine(base, embs[i])
            results.append(FilteredMutation(text=mut, similarity=round(sim, 4),
                                            kept=sim >= self.threshold))
        return results

    def keep(self, original: str, mutations: List[str]) -> List[str]:
        return [r.text for r in self.filter(original, mutations) if r.kept]
