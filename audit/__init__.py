"""Audit — append-only hash-chained decision log + tamper verifier."""

from .hash_chain import AuditEntry, GENESIS, HashChain
from .verifier import VerificationResult, verify_chain

__all__ = [
    "HashChain",
    "AuditEntry",
    "GENESIS",
    "verify_chain",
    "VerificationResult",
]
