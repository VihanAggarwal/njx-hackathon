"""Baseline — ProtectAI deberta-v3-base-prompt-injection-v2.

A widely deployed open-source prompt-injection detector (the model behind LLM
Guard's prompt-injection scanner), run via a transformers text-classification
pipeline. Requires `transformers` + `torch` and the ability to download
`protectai/deberta-v3-base-prompt-injection-v2`. If it cannot be loaded here, the
defense reports `available=False` and the harness SKIPS it (no fabricated numbers).
"""

from __future__ import annotations

from .base_defense import Defense, DefenseResult

_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


class LLMGuard(Defense):
    name = "ProtectAI deberta-v3 prompt-injection-v2"

    def __init__(self, threshold: float = 0.5, model_name: str = _MODEL):
        self.threshold = threshold
        self.model_name = model_name
        self._pipe = None
        self._error = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name,
                                  truncation=True, max_length=512, top_k=None)
        except Exception as e:  # pragma: no cover - env dependent
            self._error = str(e)
            self._pipe = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def _injection_prob(self, content: str) -> float:
        out = self._pipe(content)
        rows = out[0] if isinstance(out[0], list) else out
        for r in rows:
            # This model uses label "INJECTION" (vs "SAFE").
            if str(r["label"]).upper() in ("INJECTION", "LABEL_1"):
                return float(r["score"])
        return 0.0

    def _score(self, content: str, content_type: str) -> DefenseResult:
        if self._pipe is None:
            raise RuntimeError(f"LLM-Guard unavailable: {self._error}")
        prob = self._injection_prob(content)
        return DefenseResult(blocked=prob >= self.threshold, score=round(prob, 4),
                             name=self.name)
