"""Token usage x model price cost tracking.

Every LLM call (real or cached) is recorded here. Cached calls cost $0 of new
spend, but their token counts are still tracked so reports can show what a cold
run *would* have cost. A running estimate is printed after every N real calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class CostTracker:
    """Accumulates token usage and dollar cost across all LLM calls."""

    # model -> {"input_per_1k": float, "output_per_1k": float}
    prices: Dict[str, Dict[str, float]]
    print_every: int = 10

    real_calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Hypothetical cost if nothing had been cached (for "you saved $X" reporting).
    uncached_cost_usd: float = 0.0
    per_model: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def _price(self, model: str) -> Dict[str, float]:
        # Unknown model: don't crash a benchmark over a missing price row.
        return self.prices.get(model, {"input_per_1k": 0.0, "output_per_1k": 0.0})

    def _cost_of(self, model: str, input_tokens: int, output_tokens: int) -> float:
        p = self._price(model)
        # .get() so a partially-filled price row never crashes a benchmark.
        return (input_tokens / 1000.0) * p.get("input_per_1k", 0.0) + (
            output_tokens / 1000.0
        ) * p.get("output_per_1k", 0.0)

    def record(
        self, model: str, input_tokens: int, output_tokens: int, cached: bool
    ) -> float:
        """Record one call. Returns the *new* dollars spent (0 for cache hits)."""
        call_cost = self._cost_of(model, input_tokens, output_tokens)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.uncached_cost_usd += call_cost

        bucket = self.per_model.setdefault(
            model,
            {"real_calls": 0, "cached_calls": 0, "input_tokens": 0,
             "output_tokens": 0, "cost_usd": 0.0},
        )
        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens

        if cached:
            self.cached_calls += 1
            bucket["cached_calls"] += 1
            spent = 0.0
        else:
            self.real_calls += 1
            bucket["real_calls"] += 1
            self.cost_usd += call_cost
            bucket["cost_usd"] += call_cost
            spent = call_cost
            if self.print_every and self.real_calls % self.print_every == 0:
                self.print_running()
        return spent

    def print_running(self) -> None:
        print(
            f"[cost] {self.real_calls} live calls "
            f"({self.cached_calls} cache hits) | "
            f"in={self.input_tokens:,} out={self.output_tokens:,} tok | "
            f"spent=${self.cost_usd:.4f} "
            f"(uncached would be ${self.uncached_cost_usd:.4f})"
        )

    def summary(self) -> Dict:
        return {
            "real_calls": self.real_calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "uncached_cost_usd": round(self.uncached_cost_usd, 6),
            "saved_usd": round(self.uncached_cost_usd - self.cost_usd, 6),
            "per_model": self.per_model,
        }

    def print_final(self) -> None:
        s = self.summary()
        print("\n" + "=" * 60)
        print("LLM COST SUMMARY")
        print("=" * 60)
        print(f"  Live API calls : {s['real_calls']}")
        print(f"  Cache hits     : {s['cached_calls']}")
        print(f"  Input tokens   : {s['input_tokens']:,}")
        print(f"  Output tokens  : {s['output_tokens']:,}")
        print(f"  Total cost     : ${s['cost_usd']:.4f}")
        print(f"  Saved by cache : ${s['saved_usd']:.4f}")
        for model, b in s["per_model"].items():
            print(
                f"    - {model}: {b['real_calls']} live / {b['cached_calls']} cached, "
                f"${b['cost_usd']:.4f}"
            )
        print("=" * 60)
