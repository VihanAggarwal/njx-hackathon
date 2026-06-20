"""Audit — append-only, hash-chained decision log.

Every decision at every layer (prefilter verdict, dual-LLM outcome, taint check,
review-gate routing, KB write) is appended as an entry whose hash incorporates the
previous entry's hash:

    entry_hash = SHA256(prev_hash || canonical_json(payload))

Any tampering with a past entry breaks the chain from that point forward, which
the verifier (verifier.py) detects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

GENESIS = "0" * 64


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def _hash(prev_hash: str, payload: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode("ascii"))
    h.update(_canonical(payload).encode("utf-8"))
    return h.hexdigest()


@dataclass
class AuditEntry:
    index: int
    layer: str
    payload: Dict[str, Any]
    prev_hash: str
    entry_hash: str
    timestamp: Optional[str] = None


class HashChain:
    def __init__(self):
        self._entries: List[AuditEntry] = []

    @property
    def head(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS

    def append(
        self, layer: str, payload: Dict[str, Any], timestamp: Optional[str] = None
    ) -> AuditEntry:
        prev = self.head
        # bind the timestamp into the hashed payload so it can't be silently edited
        hashed_payload = {"layer": layer, "ts": timestamp, "data": payload}
        entry = AuditEntry(
            index=len(self._entries),
            layer=layer,
            payload=payload,
            prev_hash=prev,
            entry_hash=_hash(prev, hashed_payload),
            timestamp=timestamp,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> List[AuditEntry]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # -- persistence -------------------------------------------------------- #
    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self._entries]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "HashChain":
        chain = cls()
        chain._entries = [AuditEntry(**d) for d in data]
        return chain

    @staticmethod
    def recompute_hash(entry: AuditEntry) -> str:
        hashed_payload = {"layer": entry.layer, "ts": entry.timestamp, "data": entry.payload}
        return _hash(entry.prev_hash, hashed_payload)
