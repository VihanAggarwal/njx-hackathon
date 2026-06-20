"""Tests for the LLM layer: provider abstraction, disk cache, cost tracking."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.cache import DiskCache
from llm.cost_tracker import CostTracker
from llm.provider import MockProvider, get_provider, _synth_from_schema

PRICES = {
    "claude-opus-4-8": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "claude-haiku-4-5": {"input_per_1k": 0.00025, "output_per_1k": 0.00125},
}


def _mock(tmp_path):
    cache = DiskCache(cache_dir=str(tmp_path / "cache"))
    cost = CostTracker(prices=PRICES)
    return MockProvider(cache, cost), cache, cost


def test_cache_key_is_deterministic_and_order_independent():
    a = DiskCache.make_key({"model": "m", "prompt": "p", "x": 1, "y": 2})
    b = DiskCache.make_key({"y": 2, "x": 1, "prompt": "p", "model": "m"})
    assert a == b
    c = DiskCache.make_key({"model": "m", "prompt": "different"})
    assert a != c


def test_cache_hit_costs_nothing_and_is_logged(tmp_path):
    prov, cache, cost = _mock(tmp_path)
    prov.register("greet", "hello world")
    r1 = prov.complete("hi", model="claude-haiku-4-5", label="greet")
    assert r1.cached is False
    r2 = prov.complete("hi", model="claude-haiku-4-5", label="greet")
    assert r2.cached is True
    assert r2.text == r1.text == "hello world"
    assert cost.real_calls == 1 and cost.cached_calls == 1
    assert cost.cost_usd > 0  # the one live call cost something


def test_cost_tracker_math():
    cost = CostTracker(prices=PRICES)
    spent = cost.record("claude-opus-4-8", input_tokens=1000, output_tokens=1000, cached=False)
    # 1000/1000*0.015 + 1000/1000*0.075 = 0.09
    assert abs(spent - 0.09) < 1e-9
    assert abs(cost.cost_usd - 0.09) < 1e-9
    cost.record("claude-opus-4-8", input_tokens=1000, output_tokens=1000, cached=True)
    assert cost.cost_usd == 0.09  # cache hit added nothing
    assert cost.cached_calls == 1


def test_json_schema_parsing(tmp_path):
    prov, _, _ = _mock(tmp_path)
    schema = {
        "type": "object",
        "properties": {"label": {"type": "string"}, "score": {"type": "number"}},
        "required": ["label", "score"],
    }
    r = prov.complete("classify", model="claude-haiku-4-5", json_schema=schema, label="cls")
    assert r.json() == {"label": "mock", "score": 0.0}


def test_synth_from_schema_enum():
    schema = {"type": "object", "properties": {
        "verdict": {"type": "string", "enum": ["allow", "block"]}},
        "required": ["verdict"]}
    assert _synth_from_schema(schema) == {"verdict": "allow"}


def test_factory_defaults_to_mock_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DUALMIND_PROVIDER", raising=False)
    cfg = {"eval": {"cache_dir": str(tmp_path / "c"), "cache_llm_responses": True},
           "costs": PRICES}
    prov = get_provider(cfg)
    assert prov.name == "mock"
