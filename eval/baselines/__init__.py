"""Competitive baselines, all behind the common Defense interface."""

from .base_defense import Defense, DefenseResult
from .llm_guard import LLMGuard
from .nemo_selfcheck import NeMoStyleSelfCheck
from .prompt_guard import PromptGuard
from .regex_filter import RegexFilter
from .vanilla_selfcheck import VanillaSelfCheck


def build_baselines(provider=None, config=None):
    """Instantiate all baselines. Unavailable ones (e.g. missing HF model) are
    still returned so the harness can report them as skipped."""
    models = (config or {}).get("models", {})
    redteam_model = models.get("redteam", "claude-haiku-4-5")
    return [
        RegexFilter(),
        PromptGuard(),
        LLMGuard(),
        NeMoStyleSelfCheck(provider=provider, model=redteam_model),
        VanillaSelfCheck(provider=provider, model=redteam_model),
    ]


__all__ = [
    "Defense",
    "DefenseResult",
    "RegexFilter",
    "PromptGuard",
    "LLMGuard",
    "NeMoStyleSelfCheck",
    "VanillaSelfCheck",
    "build_baselines",
]
