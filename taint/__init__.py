"""System 4 — taint propagation + pre-tool-call taint checking."""

from .taint_checker import TaintChecker, TaintCheckResult, TaintFinding
from .taint_tracker import (
    TaintedValue,
    TaintLabel,
    TaintTracker,
    combine_labels,
)

__all__ = [
    "TaintLabel",
    "TaintedValue",
    "TaintTracker",
    "combine_labels",
    "TaintChecker",
    "TaintCheckResult",
    "TaintFinding",
]
