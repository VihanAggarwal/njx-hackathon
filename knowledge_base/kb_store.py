"""System 7 — SQLite intercept store.

Stores every intercept: sanitized content, agent, taint trace, risk score, fired
signals, human decision + note, which layer caught it, and a timestamp. Backed by
SQLite so it persists across runs.

`replay()` re-checks a previously-seen attack and shows it is now caught at the
pre-filter (fast path) rather than the dual-LLM (slow path) — concrete proof that
the system learned from the earlier interception.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS intercepts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,
    content_hash  TEXT,
    content       TEXT,
    content_type  TEXT,
    agent         TEXT,
    risk          REAL,
    signals       TEXT,
    taint_trace   TEXT,
    routing       TEXT,
    human_decision TEXT,
    note          TEXT,
    caught_by     TEXT,
    fast_path     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_hash ON intercepts(content_hash);
CREATE INDEX IF NOT EXISTS idx_agent ON intercepts(agent);
"""


def _hash(content: str) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReplayResult:
    seen_before: bool
    original_caught_by: Optional[str] = None
    original_fast_path: Optional[bool] = None
    now_verdict: Optional[str] = None
    now_fast_path: Optional[bool] = None
    learned: bool = False
    note: str = ""


class KBStore:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def add(self, record: Dict[str, Any]) -> int:
        content = record.get("content", "")
        row = (
            record.get("ts") or _now(),
            _hash(content),
            content,
            record.get("content_type", "text"),
            record.get("agent", ""),
            float(record.get("risk", 0.0)),
            json.dumps(record.get("signals", [])),
            json.dumps(record.get("taint_trace", [])),
            record.get("routing", ""),
            record.get("human_decision"),
            record.get("note", ""),
            record.get("caught_by", ""),
            1 if record.get("fast_path") else 0,
        )
        cur = self.conn.execute(
            "INSERT INTO intercepts (ts, content_hash, content, content_type, agent, "
            "risk, signals, taint_trace, routing, human_decision, note, caught_by, "
            "fast_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["signals"] = json.loads(d.get("signals") or "[]")
        d["taint_trace"] = json.loads(d.get("taint_trace") or "[]")
        d["fast_path"] = bool(d.get("fast_path"))
        return d

    def get(self, intercept_id: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM intercepts WHERE id=?", (intercept_id,)
        ).fetchone()
        return self._to_dict(row) if row else None

    def all(self) -> List[Dict[str, Any]]:
        return [self._to_dict(r) for r in self.conn.execute("SELECT * FROM intercepts")]

    def by_agent(self, agent: str) -> List[Dict[str, Any]]:
        return [self._to_dict(r) for r in self.conn.execute(
            "SELECT * FROM intercepts WHERE agent=?", (agent,))]

    def find_by_content(self, content: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM intercepts WHERE content_hash=? ORDER BY id LIMIT 1",
            (_hash(content),),
        ).fetchone()
        return self._to_dict(row) if row else None

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM intercepts").fetchone()[0]

    # ------------------------------------------------------------------ #
    def replay(self, content: str, prefilter) -> ReplayResult:
        """Re-check a seen attack: is it now caught on the fast (pre-filter) path?"""
        prior = self.find_by_content(content)
        if prior is None:
            return ReplayResult(seen_before=False, note="not previously seen")

        res = prefilter.score(content, prior.get("content_type", "text"))
        now_blocks = res.verdict in ("block", "near_miss")
        # Learned if it originally needed the slow path (dual-LLM) but the fast
        # path now blocks it.
        originally_slow = not prior.get("fast_path", False)
        learned = originally_slow and res.verdict == "block"
        return ReplayResult(
            seen_before=True,
            original_caught_by=prior.get("caught_by"),
            original_fast_path=prior.get("fast_path"),
            now_verdict=res.verdict,
            now_fast_path=res.verdict in ("block", "near_miss"),
            learned=learned,
            note="now caught on pre-filter fast path" if learned
            else "re-checked",
        )

    def close(self) -> None:
        self.conn.close()
