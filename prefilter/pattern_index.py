"""System 0.1 — pattern index of known prompt-injection signatures.

A fast (CPU, <50ms) first line of defense. Three things beyond plain regex:
  * unicode homoglyph normalization (Cyrillic/Greek lookalikes -> ASCII),
  * zero-width character detection + stripping,
  * base64 payload extraction + recursive re-scan of the decoded text.

Seed signatures are drawn from the OWASP LLM Top 10 (LLM01 Prompt Injection) and
augmented from an optional bundled signature file (prefilter/signatures.txt).

Public API:
    PatternIndex().score(content) -> PatternResult
"""

from __future__ import annotations

import base64
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

# Common confusable -> ASCII map (subset of the Unicode confusables table).
_HOMOGLYPHS = {
    # Cyrillic
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Т": "T", "В": "B",
    "М": "M", "Н": "H", "К": "K", "і": "i", "ѕ": "s", "ԁ": "d", "ո": "n",
    # Greek
    "ο": "o", "Ο": "O", "α": "a", "ν": "v", "ι": "i", "κ": "k", "ρ": "p",
    "Ι": "I", "Α": "A", "Ε": "E", "Ζ": "Z", "Η": "H", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Fullwidth latin
    "ａ": "a", "ｅ": "e", "ｉ": "i", "ｏ": "o", "ｓ": "s",
}

_ZERO_WIDTH = ["​", "‌", "‍", "⁠", "﻿", "­"]

# (name, regex, weight) — weights combine via 1 - prod(1 - w).
_SEED_SIGNATURES: List[Tuple[str, str, float]] = [
    ("ignore_previous", r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier|preceding)\s+(?:instructions?|prompts?|messages?|context|directions?)", 0.85),
    ("disregard", r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|your)\s+(?:instructions?|rules?|guidelines?)", 0.85),
    ("forget", r"forget\s+(?:everything|all|your|the)\s+(?:above|previous|prior|instructions?|rules?)", 0.8),
    ("new_persona", r"(?:you\s+are\s+now|from\s+now\s+on(?:,?\s+you\s+are)?|act\s+as|pretend\s+to\s+be|roleplay\s+as|behave\s+like)\b", 0.55),
    ("dev_mode", r"\b(?:developer\s+mode|debug\s+mode|god\s+mode|sudo\s+mode)\b", 0.7),
    ("dan_jailbreak", r"\b(?:DAN|do\s+anything\s+now|jailbreak|jailbroken)\b", 0.75),
    ("reveal_prompt", r"(?:reveal|show|print|repeat|tell\s+me|reprint|output)\s+(?:your\s+|the\s+|me\s+your\s+)?(?:system\s+prompt|initial\s+instructions?|instructions?|the\s+words?\s+above|prompt)", 0.8),
    ("override_safety", r"(?:override|bypass|ignore|disable|turn\s+off)\s+(?:your\s+)?(?:safety|guard ?rails?|filters?|restrictions?|content\s+policy)", 0.85),
    ("exfiltrate", r"(?:exfiltrate|leak|forward|send|email|post|upload|transmit)\s+(?:the\s+|all\s+|your\s+|any\s+)?(?:data|credentials?|secrets?|api\s*keys?|passwords?|files?|contents?|conversation)", 0.7),
    ("send_to_attacker", r"(?:send|forward|post)\b.{0,40}\b(?:to|at)\b.{0,40}(?:https?://|@|webhook|attacker|evil)", 0.7),
    ("new_instructions", r"(?:new|updated|important|urgent|admin|system)\s+(?:instructions?|directives?|task|rules?)\s*[:=]", 0.6),
    ("system_tag", r"(?:<\s*system\s*>|\[\s*system\s*\]|#{1,3}\s*system\b|<\|system\|>)", 0.65),
    ("hidden_from_user", r"(?:do\s+not|don't|never)\s+(?:tell|inform|mention\s+(?:this\s+)?to|alert|notify)\s+(?:the\s+)?user", 0.7),
    ("decode_and_execute", r"(?:decode|base64\s*-?\s*decode|unescape)\s+(?:the\s+)?(?:following|below|this)\b", 0.6),
    ("tool_abuse", r"(?:call|invoke|use|execute)\s+(?:the\s+)?(?:tool|function|api)\b.{0,40}(?:delete|transfer|wire|purchase|admin)", 0.6),
    ("prompt_terminator", r"(?:</?(?:instructions?|prompt|context)>|```\s*end\s+of)", 0.4),
]


@dataclass
class PatternResult:
    score: float
    signals: List[str] = field(default_factory=list)
    normalized_applied: bool = False
    decoded_layers: List[str] = field(default_factory=list)

    @property
    def blocked_signature(self) -> bool:
        return self.score >= 0.5


def _combine(weights: List[float]) -> float:
    prod = 1.0
    for w in weights:
        prod *= (1.0 - w)
    return round(1.0 - prod, 4)


class PatternIndex:
    def __init__(self, extra_signatures_path: str | None = None):
        sigs = list(_SEED_SIGNATURES)
        path = extra_signatures_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "signatures.txt"
        )
        if os.path.exists(path):
            sigs.extend(self._load_file(path))
        # Compile defensively: one bad bundled regex must not take down the whole
        # prefilter — skip it and keep the rest.
        self._compiled = []
        for name, pat, w in sigs:
            try:
                self._compiled.append(
                    (name, re.compile(pat, re.IGNORECASE | re.DOTALL), w)
                )
            except re.error as e:
                print(f"[prefilter] skipping bad signature '{name}': {e}")

    @staticmethod
    def _load_file(path: str) -> List[Tuple[str, str, float]]:
        out: List[Tuple[str, str, float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # format: name|weight|regex   (weight optional)
                parts = line.split("|", 2)
                try:
                    if len(parts) == 3:
                        out.append((parts[0], parts[2], float(parts[1])))
                    else:
                        out.append((f"bundled_{i}", line, 0.6))
                except ValueError as e:
                    print(f"[prefilter] skipping malformed signature line {i}: {e}")
        return out

    @staticmethod
    def _normalize_homoglyphs(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return "".join(_HOMOGLYPHS.get(ch, ch) for ch in text)

    @staticmethod
    def _strip_zero_width(text: str) -> Tuple[str, bool]:
        found = any(z in text for z in _ZERO_WIDTH)
        if found:
            for z in _ZERO_WIDTH:
                text = text.replace(z, "")
        return text, found

    @staticmethod
    def _looks_like_text(s: str) -> bool:
        if not s or len(s) < 8:
            return False
        printable = sum(c.isprintable() or c.isspace() for c in s)
        return printable / len(s) > 0.9 and bool(re.search(r"[A-Za-z]{4,}", s))

    @classmethod
    def _decode_run(cls, run: str, attempt_cap: int = 400) -> str | None:
        """Recover a base64 payload from a run that may be glued to other text.

        Attackers concatenate the payload with surrounding tokens (no whitespace),
        so the greedy run carries leading/trailing garbage. Slide the start offset
        (handles leading junk) and trim 4-aligned length from the end (handles
        trailing junk), returning the first window that decodes to natural text.
        """
        n = len(run)
        attempts = 0
        for start in range(0, max(1, n - 15)):
            sub = run[start:]
            L = len(sub) - (len(sub) % 4)
            while L >= 16:
                attempts += 1
                if attempts > attempt_cap:
                    return None
                try:
                    raw = base64.b64decode(sub[:L], validate=True)
                    s = raw.decode("utf-8", errors="strict")
                    if cls._looks_like_text(s):
                        return s
                except Exception:
                    pass
                L -= 4
        return None

    @classmethod
    def _extract_base64(cls, text: str) -> List[str]:
        decoded: List[str] = []
        for m in re.finditer(r"[A-Za-z0-9+/]{16,}={0,2}", text):
            s = cls._decode_run(m.group(0))
            if s is not None:
                decoded.append(s)
        return decoded

    def _scan(self, text: str) -> List[Tuple[str, float]]:
        hits: List[Tuple[str, float]] = []
        for name, rx, w in self._compiled:
            if rx.search(text):
                hits.append((name, w))
        return hits

    def score(self, content: str, content_type: str = "text") -> PatternResult:
        signals: List[str] = []
        weights: List[float] = []

        # raw scan
        for name, w in self._scan(content):
            signals.append(name)
            weights.append(w)

        # zero-width
        stripped, had_zw = self._strip_zero_width(content)
        if had_zw:
            signals.append("zero_width_chars")
            weights.append(0.6)

        # homoglyph normalization — only credit if it surfaces NEW hits
        normalized = self._normalize_homoglyphs(stripped)
        normalized_applied = normalized != stripped
        if normalized_applied:
            raw_names = {n for n, _ in self._scan(stripped)}
            for name, w in self._scan(normalized):
                if name not in raw_names and name not in signals:
                    signals.append(name)
                    weights.append(w)
            signals.append("homoglyph_obfuscation")
            weights.append(0.55)

        # base64 payloads
        decoded_layers: List[str] = []
        for dec in self._extract_base64(normalized):
            decoded_layers.append(dec)
            inner = self._scan(self._normalize_homoglyphs(dec))
            if inner:
                if "base64_payload" not in signals:
                    signals.append("base64_payload")
                    weights.append(0.7)
                for name, w in inner:
                    tag = f"b64:{name}"
                    if tag not in signals:
                        signals.append(tag)
                        weights.append(w)

        return PatternResult(
            score=_combine(weights),
            signals=signals,
            normalized_applied=normalized_applied,
            decoded_layers=decoded_layers,
        )
