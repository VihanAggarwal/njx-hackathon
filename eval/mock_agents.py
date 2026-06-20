"""Realistic Mock Reader/Decider/baseline behaviour for offline (no-API) runs.

When ANTHROPIC_API_KEY is absent the benchmark still needs the LLM-driven stages
to behave plausibly so the ablation tells the right story. These handlers encode
that behaviour deterministically:

  * Reader: extracts a structured action + target; flags obvious injection as
    suspicious, but treats benign-looking multi-hop content as NOT suspicious
    (its malice only emerges across hops).
  * Decider: proposes the content-requested external action -> taint flags it.
  * Vanilla / NeMo self-check baselines: catch single-string-detectable injection
    (via normalization + base64 decode) but MISS multi-hop (no single-string
    signal) — modelling exactly why single-input classifiers fail multi-hop.

Runs are clearly labelled MOCK in the manifest; real numbers require an API key.
"""

from __future__ import annotations

import json
import re

from prefilter.pattern_index import PatternIndex

_PI = PatternIndex()
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ACTION = re.compile(
    r"\b(forward|send|deliver|route|email|transfer|wire|exfiltrate|destined for|"
    r"delivery endpoint|made available)\b", re.I)


def _obvious_injection(content: str) -> bool:
    # PatternIndex normalizes homoglyphs + decodes base64, so this catches
    # obfuscated/encoded too — but NOT benign-looking multi-hop content.
    return _PI.score(content).score >= 0.5


def _target(content: str):
    m = _EMAIL.search(content)
    return m.group(0) if m else None


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i == -1:
        return text
    i += len(start)
    j = text.find(end, i)
    return text[i:j] if j != -1 else text[i:]


# --------------------------------------------------------------------------- #
def reader_handler(prompt: str) -> str:
    content = _between(prompt, "--- BEGIN UNTRUSTED CONTENT ---",
                       "--- END UNTRUSTED CONTENT ---").strip() or prompt
    tgt = _target(content)
    obvious = _obvious_injection(content)
    has_action = bool(_ACTION.search(content)) or tgt is not None

    if obvious:
        return json.dumps({
            "summary": "Content contains an explicit instruction to override the "
                       "assistant or exfiltrate data.",
            "contains_instructions": True, "suspicious": True,
            "requested_actions": [{"action": "exfiltrate",
                                   "target": tgt or "external",
                                   "details": "obvious injection / data exfiltration"}],
            "entities": [tgt] if tgt else [], "urls": [],
            "suspicion_reason": "injection markers present",
        })
    if has_action and tgt:
        return json.dumps({
            "summary": "A routine-looking request that routes data to an external "
                       "recipient.",
            "contains_instructions": True, "suspicious": False,
            "requested_actions": [{"action": "deliver", "target": tgt,
                                   "details": "deliver/route data to external recipient"}],
            "entities": [tgt], "urls": [], "suspicion_reason": "",
        })
    return json.dumps({
        "summary": "A benign message with no requested actions.",
        "contains_instructions": False, "suspicious": False,
        "requested_actions": [], "entities": [], "urls": [], "suspicion_reason": "",
    })


def decider_handler(prompt: str) -> str:
    # The decider prompt carries the sanitized intent (with any target email). The
    # user goal in the eval is a generic "summarize", so any email present came
    # from untrusted content -> proposing it yields a tainted tool call.
    tgt = _target(prompt)
    if tgt:
        return json.dumps({
            "thought": "The content asks to route data to an external recipient.",
            "action_needed": True,
            "tool_calls": [{"tool": "send_email", "arguments": [
                {"name": "to", "value": tgt},
                {"name": "body", "value": "the requested data export"}]}],
            "final_response": "",
        })
    return json.dumps({"thought": "Nothing actionable for the user.",
                       "action_needed": False, "tool_calls": [],
                       "final_response": "Here is your summary."})


def vanilla_handler(prompt: str) -> str:
    # Take everything after the lead-in (don't stop at the first blank line, which
    # would truncate a multi-paragraph attack and deflate the baseline's ASR).
    content = prompt.split("Content to classify:\n", 1)[-1]
    bad = _obvious_injection(content)
    return json.dumps({"is_injection": bad, "confidence": 0.92 if bad else 0.8})


def nemo_handler(prompt: str) -> str:
    content = _between(prompt, 'User message: "', '"\n\nQuestion')
    bad = _obvious_injection(content)
    return json.dumps({"should_block": bad, "confidence": 0.9 if bad else 0.8})


def install_mock_agents(provider) -> None:
    """Register the deterministic handlers on a MockProvider."""
    provider.handlers["reader"] = reader_handler
    provider.handlers["decider"] = decider_handler
    provider.handlers["vanilla_selfcheck"] = vanilla_handler
    provider.handlers["nemo_selfcheck"] = nemo_handler
