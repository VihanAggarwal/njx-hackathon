"""Baseline — ProtectAI deberta-v3-base-prompt-injection-v2.

A widely deployed open-source prompt-injection detector (the model behind LLM
Guard's prompt-injection scanner). We score it via REAL model weights, two ways:

  1. PRECOMPUTED ONNX SCORES (preferred, torch-free). A sidecar file
     `eval/results/_hf_scores_protectai.json` maps sha256(content) ->
     injection_probability, produced by `hf_onnx_runner.py` running in a clean
     standalone-Python venv where onnxruntime loads (the Anaconda-derived main
     venv cannot load torch/onnxruntime — DLL-init failure). run_benchmark wires
     this automatically when a clean venv is configured.
  2. IN-PROCESS transformers pipeline (needs torch in *this* venv).

If neither backend is available the defense reports `available=False` and the
harness SKIPS it — no fabricated numbers.
"""

from __future__ import annotations

import hashlib
import json
import os

from .base_defense import Defense, DefenseResult

_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCORES_FILE = os.path.join(_HERE, "..", "results", "_hf_scores_protectai.json")


class LLMGuard(Defense):
    name = "ProtectAI deberta-v3 prompt-injection-v2"

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
        # 2) in-process transformers pipeline (needs torch) --------------- #
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
                raise KeyError("content not in precomputed ProtectAI scores")
            return float(self._scores[h])
        out = self._pipe(content)
        rows = out[0] if isinstance(out[0], list) else out
        for r in rows:
            # This model uses label "INJECTION" (vs "SAFE").
            if str(r["label"]).upper() in ("INJECTION", "LABEL_1"):
                return float(r["score"])
        return 0.0

    def _score(self, content: str, content_type: str) -> DefenseResult:
        if not self.available:
            raise RuntimeError(f"LLM-Guard unavailable: {self._error}")
        prob = self._injection_prob(content)
        return DefenseResult(blocked=prob >= self.threshold, score=round(prob, 4),
                             name=self.name)
