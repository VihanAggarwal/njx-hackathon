"""System 4 — pre-tool-call taint scanning.

Before the Decider emits any tool call, every argument is scanned. If any token
carries DERIVED_FROM_UNTRUSTED (or UNTRUSTED), the call is flagged: untrusted
content has reached a privileged action. Arguments may be TaintedValue objects or
arbitrary nested structures (dict/list) containing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .taint_tracker import TaintedValue, TaintLabel


@dataclass
class TaintFinding:
    arg_path: str
    label: TaintLabel
    hop: int
    field_name: str
    note: str = ""


@dataclass
class TaintCheckResult:
    tool_name: str
    flagged: bool
    findings: List[TaintFinding] = field(default_factory=list)
    max_hop: int = 0

    @property
    def reason(self) -> str:
        if not self.flagged:
            return "no tainted arguments"
        paths = ", ".join(f"{f.arg_path}({f.label.value},hop={f.hop})" for f in self.findings)
        return f"tool '{self.tool_name}' received tainted arguments: {paths}"


class TaintChecker:
    """Scans tool-call arguments for tainted values."""

    def check_tool_call(self, tool_name: str, args: Dict[str, Any]) -> TaintCheckResult:
        findings: List[TaintFinding] = []
        for key, val in (args or {}).items():
            self._scan(f"args.{key}", val, findings)
        flagged = len(findings) > 0
        max_hop = max((f.hop for f in findings), default=0)
        return TaintCheckResult(
            tool_name=tool_name, flagged=flagged, findings=findings, max_hop=max_hop
        )

    def _scan(self, path: str, val: Any, findings: List[TaintFinding]) -> None:
        if isinstance(val, TaintedValue):
            if val.is_tainted:
                findings.append(
                    TaintFinding(
                        arg_path=path,
                        label=val.label,
                        hop=val.hop,
                        field_name=val.field_name,
                        note=val.note,
                    )
                )
            # also scan the wrapped value in case it nests more tainted values
            self._scan(path + ".value", val.value, findings)
        elif isinstance(val, dict):
            for k, v in val.items():
                self._scan(f"{path}.{k}", v, findings)
        elif isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                self._scan(f"{path}[{i}]", v, findings)
        # plain scalars carry no taint metadata -> ignored
