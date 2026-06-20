"""Dataset loading + caching.

For each benchmark dataset we try to load a REAL corpus from its configured cache
path; if it isn't present we generate a realistic SYNTHETIC equivalent and label it
as such (never presented as real). Provenance + a content hash are recorded per
dataset so the run manifest is reproducible.

Datasets:
  * llmail_inject  — indirect-injection email corpus (arXiv:2506.09956). Real data
                     can be dropped at datasets.llmailinja_path/data.json.
  * agentdojo      — agent security suite (arXiv:2406.13352). Real data at
                     datasets.agentdojo_path/data.json.
  * multihop       — held-out 1/2/3-hop indirect attacks, generated in-house
                     (synthetic BY DESIGN — they stress-test taint propagation).
  * benign         — benign messages for false-positive measurement.

A real cache file is a JSON list of {content, is_attack, attack_class, content_type}.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

from . import synthetic_generator as synth


def _hash_items(items: List[Dict]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it["content"].encode("utf-8"))
        h.update(str(it["is_attack"]).encode())
    return h.hexdigest()[:16]


def _load_local(path: str) -> Optional[List[Dict]]:
    """Load a real cached corpus (JSON list) if present, else None."""
    if not path:
        return None
    candidate = path if path.endswith(".json") else os.path.join(path, "data.json")
    if not os.path.exists(candidate):
        return None
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for d in data:
            items.append({
                "content": d["content"],
                "is_attack": bool(d.get("is_attack", True)),
                "attack_class": d.get("attack_class", "unknown"),
                "content_type": d.get("content_type", "text"),
                "source": "real",
            })
        return items
    except Exception:
        return None


def _filter_classes(items: List[Dict], classes) -> List[Dict]:
    return [it for it in items if it["attack_class"] in classes]


def _dataset(items: List[Dict], provenance: str, name: str) -> Dict:
    return {"name": name, "items": items, "provenance": provenance,
            "hash": _hash_items(items)}


def load_all(config: dict, seed: int = 42, n_per_class: int = 8,
             benign_n: int = 40) -> Dict[str, Dict]:
    ds_cfg = (config or {}).get("datasets", {})
    out: Dict[str, Dict] = {}

    # --- LLMail-Inject -------------------------------------------------- #
    real = _load_local(ds_cfg.get("llmailinja_path", ""))
    if real:
        out["llmail_inject"] = _dataset(real, "real", "llmail_inject")
    else:
        pool = synth.generate_attacks(n_per_class, seed=seed)
        items = _filter_classes(pool, {"direct", "indirect", "obfuscated", "encoded"})
        out["llmail_inject"] = _dataset(items, "SYNTHETIC", "llmail_inject")

    # --- AgentDojo ------------------------------------------------------ #
    real = _load_local(ds_cfg.get("agentdojo_path", ""))
    if real:
        out["agentdojo"] = _dataset(real, "real", "agentdojo")
    else:
        pool = synth.generate_attacks(n_per_class, seed=seed + 1)
        items = _filter_classes(pool, {"direct", "indirect", "jailbreak", "multi_hop_1"})
        out["agentdojo"] = _dataset(items, "SYNTHETIC", "agentdojo")

    # --- Multi-hop held-out (in-house, synthetic by design) ------------- #
    real = _load_local(ds_cfg.get("multihop_path", ""))
    if real:
        out["multihop"] = _dataset(real, "real", "multihop")
    else:
        items = synth.generate_multihop_only(max(2, n_per_class), seed=seed + 2)
        out["multihop"] = _dataset(items, "SYNTHETIC", "multihop")

    # --- Benign --------------------------------------------------------- #
    real = _load_local(ds_cfg.get("benign_path", ""))
    if real:
        out["benign"] = _dataset(real, "real", "benign")
    else:
        items = synth.generate_benign(benign_n, seed=seed + 3)
        out["benign"] = _dataset(items, "SYNTHETIC", "benign")

    return out


def combined(datasets: Dict[str, Dict]) -> List[Dict]:
    """Flatten all dataset items into one list (with a dataset tag)."""
    flat: List[Dict] = []
    for name, ds in datasets.items():
        for it in ds["items"]:
            row = dict(it)
            row["dataset"] = name
            flat.append(row)
    return flat


def manifest(datasets: Dict[str, Dict]) -> Dict:
    """Per-dataset provenance, hash, and class distribution for the run manifest."""
    m = {}
    for name, ds in datasets.items():
        classes: Dict[str, int] = {}
        for it in ds["items"]:
            classes[it["attack_class"]] = classes.get(it["attack_class"], 0) + 1
        m[name] = {
            "provenance": ds["provenance"],
            "hash": ds["hash"],
            "n": len(ds["items"]),
            "n_attacks": sum(it["is_attack"] for it in ds["items"]),
            "classes": classes,
        }
    return m


def print_summary(datasets: Dict[str, Dict]) -> None:
    print("\n### DATASET SUMMARY ###")
    total_a = total_b = 0
    any_synth = False
    for name, ds in datasets.items():
        na = sum(it["is_attack"] for it in ds["items"])
        nb = len(ds["items"]) - na
        total_a += na; total_b += nb
        any_synth = any_synth or ds["provenance"] == "SYNTHETIC"
        classes = {}
        for it in ds["items"]:
            classes[it["attack_class"]] = classes.get(it["attack_class"], 0) + 1
        tag = "[SYNTHETIC]" if ds["provenance"] == "SYNTHETIC" else "[real]"
        print(f"  {name:<14} {tag:<12} attacks={na:<4} benign={nb:<4} "
              f"hash={ds['hash']}")
        print(f"      classes: {classes}")
    print(f"  TOTAL: {total_a} attacks, {total_b} benign")
    if any_synth:
        print("  NOTE: SYNTHETIC datasets are clearly labeled and are NOT real "
              "benchmark data (see README dataset provenance).")
