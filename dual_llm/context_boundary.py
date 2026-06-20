"""System 1 — hard context boundary enforcement.

The Decider must NEVER see raw untrusted content; it only ever receives the
Reader's structured, sanitized intent. This module enforces that invariant *in
code* (not just by convention): before any Decider prompt is sent, we assert that
no meaningful span of the raw untrusted content appears in it.

A test (test_dual_llm.py) deliberately tries to leak raw content into the Decider
prompt and confirms this raises.
"""

from __future__ import annotations

import re


class ContextBoundaryViolation(Exception):
    """Raised when raw untrusted content leaks into the Decider's prompt."""


def _shingles(text: str, n: int = 8) -> set[str]:
    """Word n-gram shingles, lowercased, for robust overlap detection."""
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


class ContextBoundary:
    # 15-word shingle: long enough that a Reader legitimately quoting a short span
    # (a recipient address, a phrase) doesn't trip it, but a wholesale dump of the
    # raw payload does. For raw content shorter than this, the window adapts down
    # (see assert_no_raw_content) so short verbatim secrets are still caught.
    def __init__(self, min_shingle: int = 15):
        self.min_shingle = min_shingle

    def assert_no_raw_content(self, decider_prompt: str, raw_untrusted: str) -> None:
        """Raise if any meaningful span of raw_untrusted appears in the prompt.

        The shingle size adapts to the *raw* content length: short, high-value
        spans (account numbers, short commands, addresses) have fewer than
        `min_shingle` words and would otherwise slip past a fixed 8-gram window.
        """
        if not raw_untrusted or not decider_prompt:
            return
        raw_words = re.findall(r"\w+", raw_untrusted.lower())
        if not raw_words:
            return
        n_eff = min(self.min_shingle, len(raw_words))
        prompt_shingles = _shingles(decider_prompt, n_eff)
        for sh in _shingles(raw_untrusted, n_eff):
            if sh and sh in prompt_shingles:
                raise ContextBoundaryViolation(
                    "raw untrusted content leaked into the Decider prompt: "
                    f"...{sh[:80]}..."
                )

    def is_clean(self, decider_prompt: str, raw_untrusted: str) -> bool:
        try:
            self.assert_no_raw_content(decider_prompt, raw_untrusted)
            return True
        except ContextBoundaryViolation:
            return False
