"""Common defense interface.

Every competitor (and DUALMIND, via an adapter) exposes the same surface so they
all slot into the identical eval harness:

    Defense.score(content, content_type="text") -> DefenseResult

`score` is a confidence/risk in [0,1] used for ROC/PR/calibration; `blocked` is the
binary decision at the defense's own threshold.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class DefenseResult:
    blocked: bool
    score: float                 # confidence that content is an injection, [0,1]
    name: str
    latency_ms: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


class Defense(ABC):
    name: str = "defense"
    # label suffix, e.g. "(reimplementation)"; surfaced in all outputs
    label_suffix: str = ""
    threshold: float = 0.5

    @property
    def available(self) -> bool:
        return True

    @property
    def display_name(self) -> str:
        return f"{self.name} {self.label_suffix}".strip()

    @abstractmethod
    def _score(self, content: str, content_type: str) -> DefenseResult:
        ...

    def score(self, content: str, content_type: str = "text") -> DefenseResult:
        t0 = time.perf_counter()
        res = self._score(content, content_type)
        res.latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        res.name = self.display_name
        return res

    def score_batch(self, items: Sequence[Dict]) -> List[DefenseResult]:
        return [self.score(it["content"], it.get("content_type", "text")) for it in items]
