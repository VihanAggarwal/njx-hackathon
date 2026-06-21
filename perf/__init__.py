"""perf — latency optimization + honest latency profiling.

The pipeline's fast-path early-exit (pre-filter fast-block / fast-allow) lives in
pipeline.py; this package profiles its effect: real per-stage costs (cache off) +
the per-request path distribution -> an honest p50/p95/p99 and fast-path fraction.
"""

from .latency_profiler import (
    fast_path_fraction,
    measure_stage_latencies,
    percentiles,
    synthesize_latencies,
)

__all__ = [
    "measure_stage_latencies",
    "synthesize_latencies",
    "percentiles",
    "fast_path_fraction",
]
