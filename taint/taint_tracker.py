"""System 4 — taint propagation through the reasoning chain.

Three taint states:
    TRUSTED                  — originated from a trusted source (system/operator).
    UNTRUSTED                — came directly from untrusted content.
    DERIVED_FROM_UNTRUSTED   — computed from an UNTRUSTED or DERIVED value.

The propagation rule is monotone: any value computed from at least one
UNTRUSTED/DERIVED input inherits DERIVED_FROM_UNTRUSTED. Taint never silently
downgrades, so it survives arbitrarily long (multi-hop) reasoning chains — that
is what lets DUALMIND catch 2- and 3-hop indirect injection that single-input
classifiers structurally cannot.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


class TaintLabel(enum.Enum):
    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"
    DERIVED_FROM_UNTRUSTED = "DERIVED_FROM_UNTRUSTED"

    @property
    def is_tainted(self) -> bool:
        return self in (TaintLabel.UNTRUSTED, TaintLabel.DERIVED_FROM_UNTRUSTED)


def combine_labels(labels: Sequence[TaintLabel]) -> TaintLabel:
    """Join of taint labels under the propagation lattice.

    TRUSTED is the bottom; any tainted input lifts the result to
    DERIVED_FROM_UNTRUSTED (UNTRUSTED is reserved for *direct* sources only).
    """
    if any(l.is_tainted for l in labels):
        return TaintLabel.DERIVED_FROM_UNTRUSTED
    return TaintLabel.TRUSTED


_ids = itertools.count(1)


@dataclass
class TaintedValue:
    """A value plus its taint label and provenance (the chain that produced it)."""

    value: Any
    label: TaintLabel
    field_name: str = ""
    hop: int = 0                       # distance from the nearest untrusted source
    parents: List[int] = field(default_factory=list)
    note: str = ""
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def is_tainted(self) -> bool:
        return self.label.is_tainted


class TaintTracker:
    """Creates and propagates taint across a reasoning chain."""

    def __init__(self):
        self._values: dict[int, TaintedValue] = {}

    # -- sources ------------------------------------------------------------ #
    def trusted(self, value: Any, field_name: str = "", note: str = "") -> TaintedValue:
        return self._register(
            TaintedValue(value, TaintLabel.TRUSTED, field_name, hop=0, note=note)
        )

    def untrusted(self, value: Any, field_name: str = "", note: str = "") -> TaintedValue:
        return self._register(
            TaintedValue(value, TaintLabel.UNTRUSTED, field_name, hop=0, note=note)
        )

    def from_label(
        self, value: Any, label: TaintLabel, field_name: str = ""
    ) -> TaintedValue:
        return self._register(TaintedValue(value, label, field_name))

    # -- derivation --------------------------------------------------------- #
    def derive(
        self,
        value: Any,
        parents: Sequence[TaintedValue],
        field_name: str = "",
        note: str = "",
    ) -> TaintedValue:
        """Produce a new value computed from `parents`, propagating taint."""
        parents = list(parents)
        label = combine_labels([p.label for p in parents])
        if label.is_tainted:
            tainted_hops = [p.hop for p in parents if p.is_tainted]
            hop = (min(tainted_hops) + 1) if tainted_hops else 1
        else:
            hop = 0
        tv = TaintedValue(
            value=value,
            label=label,
            field_name=field_name,
            hop=hop,
            parents=[p.id for p in parents],
            note=note,
        )
        return self._register(tv)

    # -- provenance --------------------------------------------------------- #
    def lineage(self, tv: TaintedValue) -> List[TaintedValue]:
        """Walk the provenance chain back to its sources (root-first)."""
        seen: List[TaintedValue] = []
        stack = [tv.id]
        visited = set()
        while stack:
            vid = stack.pop()
            if vid in visited or vid not in self._values:
                continue
            visited.add(vid)
            node = self._values[vid]
            seen.append(node)
            stack.extend(node.parents)
        return list(reversed(seen))

    def get(self, value_id: int) -> Optional[TaintedValue]:
        return self._values.get(value_id)

    def _register(self, tv: TaintedValue) -> TaintedValue:
        self._values[tv.id] = tv
        return tv
