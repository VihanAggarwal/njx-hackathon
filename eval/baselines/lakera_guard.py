"""Baseline — Lakera Guard (COMMERCIAL prompt-injection detector).

The closest apples-to-apples commercial competitor: a paid, hosted guard product
purpose-built for prompt-injection / jailbreak detection. We measure it for REAL by
calling its public v2 API on the identical LLMail-Inject content; there are no
fabricated or vendor-reported numbers here.

  POST https://api.lakera.ai/v2/guard
  Authorization: Bearer $LAKERA_API_KEY
  body: {"messages":[{"role":"user","content": <email>}], "breakdown": true}
  -> top-level {"flagged": bool, "breakdown":[{detector_type, detected, ...}]}

If LAKERA_API_KEY is not set (or a call fails), the defense reports
`available=False` and the harness SKIPS it — never fabricated. A small disk cache
keeps re-runs free and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request

from .base_defense import Defense, DefenseResult

_ENDPOINT = "https://api.lakera.ai/v2/guard"
_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE = os.path.join(_HERE, "..", "results", "_lakera_cache.json")
# detector types that mean "prompt injection / attack" (vs PII, profanity, etc.)
_PI_DETECTORS = ("prompt_attack", "prompt_injection", "jailbreak", "moderated_content")


class LakeraGuard(Defense):
    name = "Lakera Guard (commercial)"

    def __init__(self, threshold: float = 0.5, api_key: str | None = None,
                 timeout: int = 30):
        self.threshold = threshold
        self.api_key = api_key or os.environ.get("LAKERA_API_KEY")
        self.timeout = timeout
        self._error = None if self.api_key else "LAKERA_API_KEY not set"
        self._cache = {}
        if os.path.exists(_CACHE):
            try:
                self._cache = json.load(open(_CACHE, "r", encoding="utf-8"))
            except Exception:  # pragma: no cover - corrupt cache
                self._cache = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
            json.dump(self._cache, open(_CACHE, "w", encoding="utf-8"))
        except Exception:  # pragma: no cover
            pass

    def _call(self, content: str) -> dict:
        body = json.dumps({"messages": [{"role": "user", "content": content}],
                           "breakdown": True}).encode("utf-8")
        req = urllib.request.Request(
            _ENDPOINT, data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _injection_prob(self, content: str) -> float:
        h = hashlib.sha256(("lakera\x00" + content).encode("utf-8")).hexdigest()
        if h in self._cache:
            return float(self._cache[h])
        data = self._call(content)
        flagged = bool(data.get("flagged", False))
        # Prefer a numeric prompt-injection score if the API returns one; else the
        # binary flag. Look through breakdown / results for a relevant detector.
        score = None
        for row in (data.get("breakdown") or []):
            dt = str(row.get("detector_type", "")).lower()
            if any(k in dt for k in _PI_DETECTORS):
                if isinstance(row.get("score"), (int, float)):
                    score = max(score or 0.0, float(row["score"]))
                elif row.get("detected"):
                    score = max(score or 0.0, 1.0)
        if score is None:
            score = 1.0 if flagged else 0.0
        self._cache[h] = score
        self._save_cache()
        return score

    def _score(self, content: str, content_type: str) -> DefenseResult:
        if not self.api_key:
            raise RuntimeError("Lakera Guard unavailable: LAKERA_API_KEY not set")
        prob = self._injection_prob(content)
        return DefenseResult(blocked=prob >= self.threshold, score=round(prob, 4),
                             name=self.name)
