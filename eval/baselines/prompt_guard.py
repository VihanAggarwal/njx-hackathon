"""Baseline — Meta Llama Prompt-Guard 86M.

Meta's open-source prompt-injection / jailbreak classifier. The official
`meta-llama/Prompt-Guard-86M` repo is GATED (needs a HF token), so like the
ProtectAI baseline we score it from REAL ONNX weights two ways:

  1. PRECOMPUTED ONNX SCORES (preferred, torch-free). A sidecar
     `eval/results/_hf_scores_promptguard.json` (sha256(content)->p_malicious),
     produced by `hf_onnx_runner.py` in the clean `.venv-ml` from an UNGATED ONNX
     mirror of the same weights (e.g. gravitee-io/Llama-Prompt-Guard-2-86M-onnx).
     run_benchmark wires this automatically when a clean venv is present.
  2. IN-PROCESS transformers pipeline (needs torch + HF access in this venv).

If neither backend is available the defense reports `available=False` and the
harness SKIPS it — it never fabricates numbers (see README).
"""

from __future__ import annotations

import hashlib
import json
import os

from .base_defense import Defense, DefenseResult

_MODEL = "meta-llama/Prompt-Guard-86M"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCORES_FILE = os.path.join(_HERE, "..", "results", "_hf_scores_promptguard.json")


class PromptGuard(Defense):
    name = "Meta Prompt-Guard-86M"

    def __init__(self, threshold: float = 0.5, model_name: str = _MODEL,
                 scores_file: str | None = None):
        self.threshold = threshold
        self.model_name = model_name
        self._pipe = None
        self._scores = None
        self._mode = None
        self._error = None
        self._load(scores_file or DEFAULT_SCORES_FILE)

    def _load(self, scores_file: str) -> None:
        # 1) precomputed ONNX scores (torch-free) -------------------------- #
        if os.path.exists(scores_file):
            try:
                blob = json.load(open(scores_file, "r", encoding="utf-8"))
                self._scores = blob.get("scores", blob)
                self.model_name = blob.get("model", self.model_name)
                self._mode = "onnx-precomputed"
                return
            except Exception as e:  # pragma: no cover - corrupt sidecar
                self._error = f"scores file unreadable: {e}"
        # 2) in-process transformers pipeline (needs torch + gated access) - #
        try:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name,
                                  truncation=True, max_length=512, top_k=None)
            self._mode = "transformers-pipeline"
        except Exception as e:  # pragma: no cover - env dependent
            self._error = (self._error + " | " if self._error else "") + str(e)
            self._pipe = None

    @property
    def available(self) -> bool:
        return self._scores is not None or self._pipe is not None

    def _injection_prob(self, content: str) -> float:
        if self._scores is not None:
            h = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if h not in self._scores:
                raise KeyError("content not in precomputed Prompt-Guard scores")
            return float(self._scores[h])
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
        if not self.available:
            raise RuntimeError(f"Prompt-Guard unavailable: {self._error}")
        prob = self._injection_prob(content)
        return DefenseResult(blocked=prob >= self.threshold, score=round(prob, 4),
                             name=self.name)
