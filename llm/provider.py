"""LLM provider abstraction — ALL LLM calls in DUALMIND route through here.

Design goals (from the build spec):
  * Provider is swappable (Anthropic today, anything tomorrow).
  * Every response is cached to disk keyed by SHA256(model + prompt + params).
  * Cost is tracked and a running estimate printed periodically.
  * No direct anthropic SDK calls anywhere else in the codebase.

API surface used (Claude Opus 4.8 / Haiku 4.5):
  * client.messages.create(...) — NOT the beta endpoint.
  * No temperature/top_p/top_k (removed on Opus 4.8 — would 400). Determinism
    comes from the disk cache, not from temperature=0.
  * Optional output_config.format (json_schema) for guaranteed-valid JSON.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .cache import DiskCache
from .cost_tracker import CostTracker


@dataclass
class LLMResponse:
    """Uniform response object returned by every provider."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost_usd: float = 0.0
    stop_reason: Optional[str] = None
    parsed: Optional[Any] = None  # populated when a json_schema was supplied

    def json(self) -> Any:
        """Return the parsed JSON body, parsing the text on demand if needed."""
        if self.parsed is not None:
            return self.parsed
        return _loads_lenient(self.text)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _loads_lenient(text: str) -> Any:
    """Parse JSON, tolerating code fences or surrounding prose."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty LLM response, cannot parse JSON")
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # last resort: grab the outermost {...} or [...]
        for open_c, close_c in (("{", "}"), ("[", "]")):
            i, j = text.find(open_c), text.rfind(close_c)
            if i != -1 and j != -1 and j > i:
                return json.loads(text[i : j + 1])
        raise


# --------------------------------------------------------------------------- #
# Base provider
# --------------------------------------------------------------------------- #
class BaseProvider(ABC):
    name: str = "base"

    def __init__(self, cache: DiskCache, cost_tracker: CostTracker):
        self.cache = cache
        self.cost = cost_tracker

    def _cache_payload(
        self, model: str, prompt: str, system: Optional[str],
        max_tokens: int, json_schema: Optional[dict],
    ) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "model": model,
            "system": system,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "json_schema": json_schema,
        }

    def complete(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        json_schema: Optional[dict] = None,
        label: str = "",
    ) -> LLMResponse:
        """Cache-checked completion. Subclasses implement ``_call`` for misses."""
        payload = self._cache_payload(model, prompt, system, max_tokens, json_schema)
        key = DiskCache.make_key(payload)

        hit = self.cache.get(key)
        if hit is not None:
            self.cost.record(model, hit["input_tokens"], hit["output_tokens"], cached=True)
            resp = LLMResponse(
                text=hit["text"], model=model,
                input_tokens=hit["input_tokens"], output_tokens=hit["output_tokens"],
                cached=True, cost_usd=0.0, stop_reason=hit.get("stop_reason"),
            )
            if json_schema is not None:
                resp.parsed = _loads_lenient(resp.text)
            return resp

        resp = self._call(prompt, model, system, max_tokens, json_schema)
        self.cache.set(key, {
            "text": resp.text,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "stop_reason": resp.stop_reason,
            "label": label,
        })
        resp.cost_usd = self.cost.record(
            model, resp.input_tokens, resp.output_tokens, cached=False
        )
        if json_schema is not None and resp.parsed is None:
            resp.parsed = _loads_lenient(resp.text)
        return resp

    @abstractmethod
    def _call(
        self, prompt: str, model: str, system: Optional[str],
        max_tokens: int, json_schema: Optional[dict],
    ) -> LLMResponse:
        ...


# --------------------------------------------------------------------------- #
# Anthropic provider
# --------------------------------------------------------------------------- #
class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, cache: DiskCache, cost_tracker: CostTracker, api_key: Optional[str] = None):
        super().__init__(cache, cost_tracker)
        try:
            import anthropic  # lazy import so the module loads without the package
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "pip install anthropic"
            ) from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _call(self, prompt, model, system, max_tokens, json_schema) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": json_schema}
            }

        try:
            resp = self.client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as e:  # pragma: no cover - network
            raise RuntimeError(
                f"Anthropic API error {getattr(e, 'status_code', '?')}: {e}"
            ) from e

        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text

        usage = resp.usage
        in_tok = getattr(usage, "input_tokens", 0) + getattr(
            usage, "cache_read_input_tokens", 0
        ) + getattr(usage, "cache_creation_input_tokens", 0)
        out_tok = getattr(usage, "output_tokens", 0)

        return LLMResponse(
            text=text, model=model, input_tokens=in_tok, output_tokens=out_tok,
            cached=False, stop_reason=getattr(resp, "stop_reason", None),
        )


# --------------------------------------------------------------------------- #
# Mock provider (offline tests + deterministic CI)
# --------------------------------------------------------------------------- #
class MockProvider(BaseProvider):
    """Deterministic, network-free provider for unit tests.

    Resolves a response in priority order:
      1. an exact match in ``responses`` keyed by the ``label``;
      2. a registered handler in ``handlers`` (label -> callable(prompt)->str);
      3. a minimal valid instance synthesized from ``json_schema``;
      4. a canned text echo.
    Token counts are estimated by word count so cost tracking still exercises.
    """

    name = "mock"

    def __init__(self, cache: DiskCache, cost_tracker: CostTracker):
        super().__init__(cache, cost_tracker)
        self.responses: Dict[str, str] = {}
        self.handlers: Dict[str, Any] = {}
        self.call_log: List[Dict[str, Any]] = []

    def register(self, label: str, value: Any) -> None:
        self.responses[label] = value if isinstance(value, str) else json.dumps(value)

    def _call(self, prompt, model, system, max_tokens, json_schema) -> LLMResponse:
        self.call_log.append({"label": "", "model": model, "prompt": prompt})
        text = self._resolve("", prompt, json_schema)
        in_tok = max(1, len((system or "").split()) + len(prompt.split()))
        out_tok = max(1, len(text.split()))
        return LLMResponse(
            text=text, model=model, input_tokens=in_tok, output_tokens=out_tok,
            cached=False, stop_reason="end_turn",
        )

    # complete() passes label down via _cache_payload only; expose it to _resolve
    def complete(self, prompt, model, system=None, max_tokens=2048,
                 json_schema=None, label=""):
        # Override to thread the label through to mock resolution.
        self._pending_label = label
        return super().complete(prompt, model, system, max_tokens, json_schema, label)

    def _resolve(self, _label: str, prompt: str, json_schema: Optional[dict]) -> str:
        label = getattr(self, "_pending_label", "") or _label
        if label in self.responses:
            return self.responses[label]
        if label in self.handlers:
            out = self.handlers[label](prompt)
            return out if isinstance(out, str) else json.dumps(out)
        if json_schema is not None:
            return json.dumps(_synth_from_schema(json_schema))
        return f"[mock:{label or 'response'}] {prompt[:80]}"


def _synth_from_schema(schema: dict) -> Any:
    """Build a minimal valid instance from a (subset of) JSON Schema."""
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if t == "object":
        props = schema.get("properties", {})
        required = schema.get("required", list(props.keys()))
        return {k: _synth_from_schema(props[k]) for k in required if k in props}
    if t == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [_synth_from_schema(item_schema)]
    if t == "string":
        return "mock"
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    if t == "null":
        return None
    return None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_provider(
    config: dict,
    cache: Optional[DiskCache] = None,
    cost_tracker: Optional[CostTracker] = None,
    force: Optional[str] = None,
) -> BaseProvider:
    """Build a provider from config + environment.

    Selection: ``force`` arg > ``DUALMIND_PROVIDER`` env > 'anthropic' if an API
    key is present > 'mock'. This lets unit tests run with no key (mock) while
    real runs use Anthropic automatically.
    """
    eval_cfg = config.get("eval", {})
    if cache is None:
        cache = DiskCache(
            cache_dir=eval_cfg.get("cache_dir", ".llm_cache/"),
            enabled=eval_cfg.get("cache_llm_responses", True),
        )
    if cost_tracker is None:
        cost_tracker = CostTracker(prices=config.get("costs", {}))

    choice = force or os.environ.get("DUALMIND_PROVIDER")
    if choice is None:
        choice = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock"

    if choice == "anthropic":
        return AnthropicProvider(cache, cost_tracker)
    if choice == "mock":
        return MockProvider(cache, cost_tracker)
    raise ValueError(f"Unknown provider: {choice}")
