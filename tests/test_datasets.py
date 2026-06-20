"""Tests for dataset loading + synthetic generation.

(Not in the spec's enumerated test list, added for coverage.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.datasets import loader, synthetic_generator as synth

CFG = {"datasets": {}}  # no local cache -> synthetic


def test_generate_attacks_covers_all_classes():
    items = synth.generate_attacks(n_per_class=3, seed=42)
    classes = {it["attack_class"] for it in items}
    assert {"direct", "indirect", "multi_hop_1", "multi_hop_2", "multi_hop_3",
            "obfuscated", "encoded", "jailbreak"} <= classes
    assert all(it["is_attack"] for it in items)
    assert all(it["source"] == "SYNTHETIC" for it in items)


def test_generate_benign():
    items = synth.generate_benign(10, seed=43)
    assert len(items) == 10
    assert all(not it["is_attack"] for it in items)


def test_multihop_only_has_three_hop_levels():
    items = synth.generate_multihop_only(2, seed=44)
    classes = {it["attack_class"] for it in items}
    assert classes == {"multi_hop_1", "multi_hop_2", "multi_hop_3"}


def test_load_all_falls_back_to_synthetic_and_is_labeled():
    ds = loader.load_all(CFG, seed=42, n_per_class=4, benign_n=12)
    assert set(ds) == {"llmail_inject", "agentdojo", "multihop", "benign"}
    for name, d in ds.items():
        assert d["provenance"] == "SYNTHETIC"  # no local cache present
        assert d["hash"]
    flat = loader.combined(ds)
    assert any(r["is_attack"] for r in flat)
    assert any(not r["is_attack"] for r in flat)
    man = loader.manifest(ds)
    assert man["benign"]["n_attacks"] == 0


def test_load_all_is_reproducible():
    a = loader.load_all(CFG, seed=7, n_per_class=4)
    b = loader.load_all(CFG, seed=7, n_per_class=4)
    assert loader.manifest(a) == loader.manifest(b)  # same seed -> same hashes
