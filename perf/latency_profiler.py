"""perf — honest, uncached latency profiling + fast-path analysis.

The eval harness caches LLM responses for determinism and cost, which makes the
measured per-request latency unrealistically small (cache hits are ~ms). This
module recovers the REAL cost by:

  1. measuring real per-stage latency (prefilter / Reader / Decider) on a small
     sample with the LLM cache DISABLED, then
  2. combining it with the per-request PATH distribution from the full run to
     synthesize an honest p50/p95/p99.

The headline it produces: most traffic exits at the pre-filter (fast path) and
never pays the dual-LLM tail, so p50 stays small even though p95/p99 carry the
real LLM cost. Adding security layers does NOT proportionally add latency.
"""

from __future__ import annotations

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics as M
from llm.cache import DiskCache
from llm.cost_tracker import CostTracker
from llm.provider import get_provider
from pipeline import DualMind, DualMindConfig


def _median(xs):
    return round(statistics.median(xs), 3) if xs else 0.0


def measure_stage_latencies(cfg, prefilter, calibrator, full_sample, fast_sample,
                            user_goal):
    """Real per-stage latency (ms) measured with the LLM cache DISABLED."""
    prov = get_provider(
        cfg,
        cache=DiskCache(cache_dir=cfg.get("eval", {}).get("cache_dir", ".llm_cache/"),
                        enabled=False, verbose=False),
        cost_tracker=CostTracker(prices=cfg.get("costs", {})),
    )
    if prov.name == "mock":
        from eval.mock_agents import install_mock_agents
        install_mock_agents(prov)
    dm = DualMind(cfg, provider=prov, prefilter=prefilter, calibrator=calibrator,
                  flags=DualMindConfig(True, True, True, True, "_latency", True))
    # Reader/Decider real cost: time the FULL-path samples.
    rd_ms, dc_ms = [], []
    for it in full_sample:
        tr = dm.process(it["content"], user_goal=user_goal,
                        content_type=it.get("content_type", "text"))
        if "reader_ms" in tr.timings:
            rd_ms.append(tr.timings["reader_ms"])
        if "decider_ms" in tr.timings:
            dc_ms.append(tr.timings["decider_ms"])
    # Fast-path real cost = a COLD pre-filter scan (cache off, unique content) — no
    # LLM. Measured separately so a benign that happens to take the full path can't
    # pollute it.
    import time as _t
    from prefilter import PreFilter
    cold_cfg = dict(cfg)
    cold_cfg["perf"] = {**cfg.get("perf", {}), "prefilter_cache": False}
    cold_pf = PreFilter(cold_cfg, classifier=getattr(prefilter, "classifier", None))
    pf_ms = []
    for i, it in enumerate(list(fast_sample) + list(full_sample)):
        t = _t.perf_counter()
        cold_pf.score(f"u{i} " + it["content"], it.get("content_type", "text"))
        pf_ms.append((_t.perf_counter() - t) * 1000.0)
    return {
        "prefilter_ms": _median(pf_ms),
        "reader_ms": _median(rd_ms),
        "decider_ms": _median(dc_ms),
        "fast_path_ms": _median(pf_ms),
    }


def synthesize_latencies(paths, stage):
    """Per-request latency from each request's PATH + the real stage costs."""
    fast = stage["fast_path_ms"] or stage["prefilter_ms"]
    full = stage["prefilter_ms"] + stage["reader_ms"] + stage["decider_ms"]
    return [full if p == "full_dual_llm" else fast for p in paths]


def percentiles(lat):
    return {
        "p50": round(M.percentile(lat, 50), 2),
        "p95": round(M.percentile(lat, 95), 2),
        "p99": round(M.percentile(lat, 99), 2),
        "mean": round(sum(lat) / len(lat), 2) if lat else 0.0,
    }


def fast_path_fraction(paths):
    if not paths:
        return 0.0
    return round(sum(1 for p in paths if p != "full_dual_llm") / len(paths), 3)
