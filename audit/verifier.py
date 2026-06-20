"""Audit — tamper-detection verifier.

Walks a HashChain and confirms (a) each entry's stored hash equals the recomputed
hash of its payload, and (b) each entry's prev_hash matches the previous entry's
hash. Any altered entry breaks both checks from that index forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .hash_chain import GENESIS, HashChain


@dataclass
class VerificationResult:
    valid: bool
    broken_index: int = -1
    reason: str = "ok"
    checked: int = 0

    def __bool__(self) -> bool:
        return self.valid


def verify_chain(
    chain: HashChain, expected_length: Optional[int] = None
) -> VerificationResult:
    # Tail-truncation leaves a valid prefix; pass the recorded length (a head/length
    # commitment kept out-of-band) to detect dropped trailing entries.
    if expected_length is not None and len(chain) != expected_length:
        return VerificationResult(
            False, len(chain),
            f"length mismatch: have {len(chain)}, expected {expected_length} "
            "(possible truncation)", len(chain),
        )
    prev = GENESIS
    for i, entry in enumerate(chain.entries):
        if entry.index != i:
            return VerificationResult(False, i, f"index mismatch at {i}", i)
        if entry.prev_hash != prev:
            return VerificationResult(
                False, i, f"prev_hash mismatch at index {i} (chain broken)", i
            )
        recomputed = HashChain.recompute_hash(entry)
        if recomputed != entry.entry_hash:
            return VerificationResult(
                False, i, f"payload tampering detected at index {i}", i
            )
        prev = entry.entry_hash
    return VerificationResult(True, -1, "ok", len(chain.entries))
