"""SHA256(model + prompt + params) -> response disk cache.

Every LLM response is cached to disk keyed by a hash of the full request. Re-runs
are free and deterministic, and cache hits are logged. Stored as one JSON file per
key under ``cache_dir``.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


class DiskCache:
    """A tiny content-addressed JSON cache on disk."""

    def __init__(self, cache_dir: str, enabled: bool = True, verbose: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.verbose = verbose
        self.hits = 0
        self.misses = 0
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        """Deterministic SHA256 over a request payload (sort_keys for stability)."""
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not os.path.exists(path):
            self.misses += 1
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt cache entry — treat as a miss and let it be overwritten.
            self.misses += 1
            return None
        self.hits += 1
        if self.verbose:
            print(f"[cache] HIT {key[:12]}")
        return value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=0)
        os.replace(tmp, path)  # atomic write

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
