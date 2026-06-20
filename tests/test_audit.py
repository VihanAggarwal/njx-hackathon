"""Tests for the audit hash-chain + tamper verifier."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audit import GENESIS, HashChain, verify_chain


def _build_chain():
    c = HashChain()
    c.append("prefilter", {"verdict": "pass", "risk": 0.1}, timestamp="t0")
    c.append("dual_llm", {"decision": "allow"}, timestamp="t1")
    c.append("review_gate", {"routing": "auto_allow"}, timestamp="t2")
    return c


def test_chain_links_and_verifies():
    c = _build_chain()
    assert len(c) == 3
    assert c.entries[0].prev_hash == GENESIS
    assert c.entries[1].prev_hash == c.entries[0].entry_hash
    res = verify_chain(c)
    assert res.valid
    assert res.checked == 3


def test_tampering_payload_is_detected():
    c = _build_chain()
    # An attacker edits a past decision (block -> allow) but can't recompute hashes.
    c.entries[0].payload["verdict"] = "block"
    res = verify_chain(c)
    assert not res.valid
    assert res.broken_index == 0
    assert "tampering" in res.reason


def test_tampering_midchain_detected():
    c = _build_chain()
    c.entries[1].payload["decision"] = "block"
    res = verify_chain(c)
    assert not res.valid
    assert res.broken_index == 1


def test_reorder_or_prevhash_edit_detected():
    c = _build_chain()
    # splice out an entry's link
    c.entries[2].prev_hash = "deadbeef" * 8
    res = verify_chain(c)
    assert not res.valid
    assert res.broken_index == 2


def test_roundtrip_serialization():
    c = _build_chain()
    data = c.to_list()
    c2 = HashChain.from_list(data)
    assert verify_chain(c2).valid
    assert c2.head == c.head
