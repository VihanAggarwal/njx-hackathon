"""Baseline — Meta Prompt-Guard-86M.

Meta's open-source prompt-injection / jailbreak classifier, run via a transformers
text-classification pipeline. Requires `transformers` + `torch` and the ability to
download `meta-llama/Prompt-Guard-86M` (a gated HF model). If the model cannot be
loaded in this environment, the defense reports `available=False` and the harness
SKIPS it — it never fabricates numbers (see README).
"""

from __future__ import annotations

from .base_defense import Defense, DefenseResult

_MODEL = "meta-llama/Prompt-Guard-86M"


class PromptGuard(Defense):
    name = "Meta Prompt-Guard-86M"

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
        # top_k=None returns a list of {label, score}; sum non-benign labels.
        rows = out[0] if isinstance(out[0], list) else out
        prob = 0.0
        for r in rows:
            label = str(r["label"]).upper()
            if label in ("JAILBREAK", "INJECTION", "MALICIOUS", "LABEL_1", "LABEL_2"):
                prob += float(r["score"])
        return min(1.0, prob)

    def _score(self, content: str, content_type: str) -> DefenseResult:
        if self._pipe is None:
            raise RuntimeError(f"Prompt-Guard unavailable: {self._error}")
        prob = self._injection_prob(content)
        return DefenseResult(blocked=prob >= self.threshold, score=round(prob, 4),
                             name=self.name)
