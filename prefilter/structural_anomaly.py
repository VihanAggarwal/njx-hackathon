"""System 0.3 — structural anomaly scoring.

Checks content against an expected-structure model for its *claimed* type. The
core intuition: passive data (an email, a web page, a search result, a calendar
invite, a product review) should not contain second-person directives aimed at an
assistant, role assignments, or tool-invocation language. When it does, that's a
strong structural signal of an indirect prompt-injection payload.

For content types where instructions ARE expected (a direct user query / chat
turn), the same features carry little weight.

Public API:
    StructuralAnomaly().score(content, content_type) -> AnomalyResult
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

# Content types that are *passive data* — instructions here are suspicious.
_PASSIVE_TYPES = {
    "email", "document", "webpage", "web_page", "html", "search_result",
    "search_results", "calendar", "calendar_invite", "review", "comment",
    "file", "pdf", "tool_output", "retrieved", "knowledge", "attachment",
}
# Content types where instructions are normal.
_ACTIVE_TYPES = {"user_query", "query", "chat", "user", "prompt", "instruction"}

_DIRECTIVE_PATTERNS: List[tuple] = [
    ("imperative_to_ai", r"\b(?:you\s+must|you\s+should|you\s+are\s+required|please\s+(?:now\s+)?(?:ignore|disregard|send|forward|delete|execute|run|print|reveal))\b", 0.5),
    ("second_person_ai", r"\b(?:you\s+are\s+(?:an?\s+)?(?:ai|assistant|model|agent|llm|chatbot)|your\s+(?:instructions?|system\s+prompt|guidelines?|task\s+is))\b", 0.6),
    ("role_assignment", r"\b(?:act\s+as|pretend\s+to\s+be|you\s+are\s+now|from\s+now\s+on|roleplay)\b", 0.55),
    ("tool_language", r"\b(?:call|invoke|execute|run)\s+(?:the\s+)?(?:tool|function|api|command|shell)\b", 0.5),
    ("instruction_marker", r"(?:^|\n)\s*(?:system|assistant|instructions?|important|note\s+to\s+(?:ai|assistant))\s*[:>\]]", 0.45),
    ("override_language", r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,30}\b(?:previous|prior|above|instructions?|rules?|safety)\b", 0.65),
    ("secrecy", r"\b(?:do\s+not|don't|never)\s+(?:tell|inform|mention\s+to|alert)\s+the\s+user\b", 0.6),
]


@dataclass
class AnomalyResult:
    score: float
    reasons: List[str] = field(default_factory=list)
    content_type: str = "text"
    expected_passive: bool = False
    feature_hits: Dict[str, int] = field(default_factory=dict)


class StructuralAnomaly:
    def __init__(self):
        self._compiled = [
            (name, re.compile(pat, re.IGNORECASE | re.MULTILINE), w)
            for name, pat, w in _DIRECTIVE_PATTERNS
        ]

    def score(self, content: str, content_type: str = "text") -> AnomalyResult:
        ctype = (content_type or "text").lower().strip()
        passive = ctype in _PASSIVE_TYPES or (
            ctype not in _ACTIVE_TYPES and ctype != "text"
        )
        # "text" (unknown provenance) is treated as mildly passive.
        weight_scale = 1.0 if passive else (0.15 if ctype in _ACTIVE_TYPES else 0.5)

        reasons: List[str] = []
        feature_hits: Dict[str, int] = {}
        weights: List[float] = []
        for name, rx, w in self._compiled:
            matches = rx.findall(content)
            if matches:
                feature_hits[name] = len(matches)
                reasons.append(name)
                # repeated directives compound, but saturate.
                eff = min(0.95, w * (1.0 + 0.15 * (len(matches) - 1)))
                weights.append(eff * weight_scale)

        # structural mismatch: a "passive" doc that is mostly imperative sentences
        if passive and content:
            imperative_ratio = _imperative_ratio(content)
            if imperative_ratio > 0.4:
                reasons.append("high_imperative_density")
                weights.append(min(0.6, imperative_ratio) * weight_scale)

        prod = 1.0
        for w in weights:
            prod *= (1.0 - w)
        score = round(1.0 - prod, 4)

        return AnomalyResult(
            score=score,
            reasons=reasons,
            content_type=ctype,
            expected_passive=passive,
            feature_hits=feature_hits,
        )


_IMPERATIVE_START = re.compile(
    r"^\s*(?:ignore|disregard|forget|send|forward|delete|execute|run|print|reveal|"
    r"do|don't|never|always|act|pretend|use|call|invoke|stop|start|tell|give|make|"
    r"output|respond|reply|write|update|change|set)\b",
    re.IGNORECASE,
)


def _imperative_ratio(content: str) -> float:
    sentences = re.split(r"(?<=[.!?\n])\s+", content.strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    imperative = sum(1 for s in sentences if _IMPERATIVE_START.match(s))
    return imperative / len(sentences)
