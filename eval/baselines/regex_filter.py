"""Baseline — regex/keyword floor filter.

The simplest possible defense: a keyword/regex blocklist. This is the floor that
shows what you get with zero ML. Always available.
"""

from __future__ import annotations

import re

from .base_defense import Defense, DefenseResult

_KEYWORDS = [
    r"ignore (all |the )?(previous|prior|above) (instructions|directions|prompts)",
    r"disregard (your |the )?(previous|prior|above)",
    r"forget (everything|all|the above)",
    r"system prompt",
    r"developer mode",
    r"\bDAN\b",
    r"do anything now",
    r"jailbreak",
    r"reveal your (instructions|prompt)",
    r"you are now",
    r"act as",
    r"bypass (the )?(safety|filter|guard)",
    r"exfiltrate",
    r"send .* to .*@",
    r"api keys?",
    r"do not tell the user",
]


class RegexFilter(Defense):
    name = "Regex/keyword filter"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._patterns = [re.compile(k, re.IGNORECASE) for k in _KEYWORDS]

    def _score(self, content: str, content_type: str) -> DefenseResult:
        hits = sum(1 for p in self._patterns if p.search(content))
        # map hit count to a [0,1] confidence: 1 hit -> 0.6, saturating.
        score = 0.0 if hits == 0 else min(1.0, 0.45 + 0.2 * hits)
        return DefenseResult(blocked=score >= self.threshold, score=round(score, 4),
                             name=self.name, meta={"hits": hits})
