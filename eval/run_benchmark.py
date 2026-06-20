"""Eval harness entrypoint — full ablation matrix + competitive baseline comparison.

Usage:
    python eval/run_benchmark.py --max-attacks 20        # cheap smoke run (default)
    python eval/run_benchmark.py --full                  # larger run
    python eval/run_benchmark.py --max-attacks 5         # plumbing check

Produces (under eval/results/):
    results.json        ablation metrics + CIs + per-class ASR + calibration + ROC/PR
    results.csv         tidy per-attack rows (every config x item)
    competitive.json    baseline + DUALMIND head-to-head metrics + CIs
    competitive.csv     tidy competitive rows
    self_hardening.json ASR-over-rounds curve
    manifest.json       seed, models, dataset hashes/provenance, provider mode, timestamp

Graphs are regenerated separately from these files (see eval/graphs.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_and_seed
from eval import metrics as M
from eval.baselines import build_baselines
from eval.datasets import loader
from eval.mock_agents import install_mock_agents
from llm import get_provider
from pipeline import DualMind, DualMindConfig
from prefilter import PreFilter
from prefilter.embedding_classifier import EmbeddingClassifier, bundled_training_data
from redteam import MutationEngine, SelfHardeningLoop, SemanticFilter

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
USER_GOAL = "Summarize this content for me."


# --------------------------------------------------------------------------- #
# Ablation matrix (the 6 configurations)
# --------------------------------------------------------------------------- #
def ablation_flags():
    return [
        DualMindConfig(False, False, False, False, "1_no_defense"),
        DualMindConfig(True, False, False, False, "2_prefilter_only"),
        DualMindConfig(False, True, False, False, "3_dual_llm_only"),
        DualMindConfig(False, True, True, False, "4_dual_llm_taint"),
        DualMindConfig(True, True, True, True, "5_full_pre_hardening"),
        DualMindConfig(True, True, True, True, "6_full_post_hardening"),
    ]


# --------------------------------------------------------------------------- #
def stratified_cap(attacks, max_attacks, seed):
    """Cap to `max_attacks` while keeping every attack class represented."""
    import random
    if max_attacks is None or len(attacks) <= max_attacks:
        return attacks
    rng = random.Random(seed)
    by_class = {}
    for a in attacks:
        by_class.setdefault(a["attack_class"], []).append(a)
    for v in by_class.values():
        rng.shuffle(v)
    chosen, classes = [], list(by_class)
    i = 0
    while len(chosen) < max_attacks and any(by_class.values()):
        cls = classes[i % len(classes)]
        if by_class[cls]:
            chosen.append(by_class[cls].pop())
        i += 1
    return chosen[:max_attacks]


def evaluate_config(flags, provider, prefilter, config, items):
    dm = DualMind(config, provider=provider,
                  prefilter=prefilter if flags.use_prefilter else None, flags=flags)
    cost_before = getattr(provider, "cost", None)
    c0 = cost_before.uncached_cost_usd if cost_before else 0.0
    results = []
    for it in items:
        tr = dm.process(it["content"], user_goal=USER_GOAL,
                        content_type=it.get("content_type", "text"),
                        ground_truth="attack" if it["is_attack"] else "benign")
        results.append({
            "is_attack": it["is_attack"], "blocked": tr.blocked, "score": tr.risk,
            "attack_class": it["attack_class"], "latency_ms": tr.latency_ms,
            "dataset": it.get("dataset", ""), "caught_by": tr.caught_by,
        })
    c1 = cost_before.uncached_cost_usd if cost_before else 0.0
    cost_per_req = (c1 - c0) / max(1, len(items))
    return results, cost_per_req


def evaluate_defense(defense, items):
    results = []
    for it in items:
        r = defense.score(it["content"], it.get("content_type", "text"))
        results.append({
            "is_attack": it["is_attack"], "blocked": r.blocked, "score": r.score,
            "attack_class": it["attack_class"], "latency_ms": r.latency_ms,
            "dataset": it.get("dataset", ""),
        })
    return results


def metrics_block(results, n_resamples, seed, cost_per_req=0.0):
    scores = [r["score"] for r in results]
    labels = [1 if r["is_attack"] else 0 for r in results]
    s = M.summary(results)
    s.update({
        "asr_ci": M.bootstrap_asr_ci(results, n_resamples, seed=seed),
        "fpr_ci": M.bootstrap_fpr_ci(results, n_resamples, seed=seed),
        "per_class_asr": M.per_class_asr(results),
        "latency": M.latency_stats(results),
        "confusion": M.confusion(results),
        "ece": M.expected_calibration_error(scores, labels),
        "reliability": M.reliability_curve(scores, labels),
        "roc": M.roc_points(scores, labels),
        "pr": M.pr_points(scores, labels),
        "cost_per_1000_usd": round(cost_per_req * 1000.0, 4),
    })
    return s


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="DUALMIND benchmark")
    ap.add_argument("--max-attacks", type=int, default=None,
                    help="cap total attacks (smoke run). Default: config eval.max_attacks")
    ap.add_argument("--full", action="store_true", help="use a larger attack/benign pool")
    ap.add_argument("--rounds", type=int, default=None, help="self-hardening rounds")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_and_seed()
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    eval_cfg = cfg.get("eval", {})
    max_attacks = args.max_attacks if args.max_attacks is not None else eval_cfg.get("max_attacks", 20)
    n_resamples = eval_cfg.get("bootstrap_resamples", 1000)
    rounds = args.rounds if args.rounds is not None else eval_cfg.get("redteam_rounds", 10)

    provider = get_provider(cfg)
    mock = provider.name == "mock"
    if mock:
        install_mock_agents(provider)
    print(f"### DUALMIND BENCHMARK  (provider={provider.name.upper()}"
          f"{' / MOCK numbers' if mock else ''}, seed={seed}, max_attacks={max_attacks}) ###")

    # --- data ----------------------------------------------------------- #
    n_per_class = 12 if args.full else 8
    benign_n = 80 if args.full else 40
    datasets = loader.load_all(cfg, seed=seed, n_per_class=n_per_class, benign_n=benign_n)
    loader.print_summary(datasets)
    flat = loader.combined(datasets)
    attacks = [r for r in flat if r["is_attack"]]
    benign = [r for r in flat if not r["is_attack"]]
    attacks = stratified_cap(attacks, max_attacks, seed)
    benign = benign[: max(max_attacks, 20)]
    items = attacks + benign
    print(f"\nEvaluating on {len(attacks)} attacks + {len(benign)} benign = {len(items)} items")

    # --- prefilter classifiers (base + hardened via self-hardening) ----- #
    print("\n### SELF-HARDENING (red-team loop) ###")
    base_clf = EmbeddingClassifier(block_threshold=cfg["thresholds"]["prefilter_block"],
                                   near_miss_threshold=cfg["thresholds"]["prefilter_near_miss"])
    bt_texts, bt_labels = bundled_training_data()
    base_clf.fit(bt_texts, bt_labels)
    base_prefilter = PreFilter(cfg, classifier=base_clf)

    # Harden a copy on a SEPARATE red-team pool (not the eval set) to avoid leakage.
    from eval.datasets import synthetic_generator as synth
    redteam_pool = [a["content"] for a in synth.generate_attacks(8, seed=seed + 777)]
    loop = SelfHardeningLoop(
        classifier=EmbeddingClassifier(
            block_threshold=cfg["thresholds"]["prefilter_block"],
            near_miss_threshold=cfg["thresholds"]["prefilter_near_miss"]),
        mutation_engine=MutationEngine(provider=None),
        semantic_filter=SemanticFilter(threshold=0.0),
        block_threshold=cfg["thresholds"]["prefilter_block"],
        init_pos=[t for t, y in zip(bt_texts, bt_labels) if y == 1],
        init_neg=[t for t, y in zip(bt_texts, bt_labels) if y == 0],
    )
    history = loop.run(redteam_pool, rounds=rounds, max_mutations=8)
    hardened_prefilter = PreFilter(cfg, classifier=loop.classifier)
    print(f"  ASR over {rounds} rounds: {history.asr_per_round}")
    print(f"  initial ASR={history.initial_asr}  final ASR={history.final_asr}")

    # --- run the ablation matrix --------------------------------------- #
    print("\n### ABLATION MATRIX ###")
    configs = {}
    per_attack_rows = []
    for flags in ablation_flags():
        pf = hardened_prefilter if flags.name.startswith("6_") else base_prefilter
        results, cost = evaluate_config(flags, provider, pf, cfg, items)
        configs[flags.name] = metrics_block(results, n_resamples, seed, cost)
        for r in results:
            per_attack_rows.append({"config": flags.name, **r})
        m = configs[flags.name]
        print(f"  {flags.name:<24} ASR={m['asr']:.3f} {tuple(m['asr_ci'])}  "
              f"FPR={m['fpr']:.3f}  F1={m['f1']:.3f}  p95={m['latency']['p95']}ms")

    # --- baselines ------------------------------------------------------ #
    competitive = {}
    if not args.no_baselines:
        print("\n### COMPETITIVE BASELINES ###")
        for d in build_baselines(provider=provider, config=cfg):
            if not d.available:
                competitive[d.display_name] = {"available": False,
                                               "reason": "model/dependency unavailable in this environment"}
                print(f"  {d.display_name:<42} SKIPPED (unavailable - not fabricated)")
                continue
            results = evaluate_defense(d, items)
            competitive[d.display_name] = {"available": True,
                                           **metrics_block(results, n_resamples, seed)}
            for r in results:
                per_attack_rows.append({"config": f"baseline:{d.display_name}", **r})
            m = competitive[d.display_name]
            print(f"  {d.display_name:<42} ASR={m['asr']:.3f} {tuple(m['asr_ci'])}  "
                  f"FPR={m['fpr']:.3f}")

    # Add DUALMIND (full post-hardening) into the competitive set for the leaderboard.
    competitive["DUALMIND (full, post-hardening)"] = {"available": True,
                                                      **configs["6_full_post_hardening"]}

    # --- manifest ------------------------------------------------------- #
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "provider": provider.name,
        "mode": "MOCK" if mock else "live",
        "models": cfg.get("models", {}),
        "max_attacks": max_attacks,
        "n_items": len(items),
        "bootstrap_resamples": n_resamples,
        "redteam_rounds": rounds,
        "datasets": loader.manifest(datasets),
    }

    # --- save ----------------------------------------------------------- #
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _save_json("results.json", {"manifest": manifest, "configs": configs,
                                "self_hardening": _hardening_dict(history)})
    _save_json("competitive.json", {"manifest": manifest, "defenses": competitive})
    _save_json("self_hardening.json", _hardening_dict(history))
    _save_json("manifest.json", manifest)
    _save_csv("results.csv", per_attack_rows,
              ["config", "dataset", "attack_class", "is_attack", "blocked", "score",
               "latency_ms", "caught_by"])
    _save_competitive_csv(competitive)

    _print_leaderboard(competitive)
    if hasattr(provider, "cost"):
        provider.cost.print_final()
    print(f"\nSaved results to {RESULTS_DIR}")
    print("Regenerate graphs with:  python eval/graphs.py --from-results eval/results/results.json")
    return 0


def _hardening_dict(history):
    return {
        "asr_per_round": history.asr_per_round,
        "bypasses_per_round": history.bypasses_per_round,
        "mutations_fired_per_round": history.mutations_fired_per_round,
        "mean_reward_per_round": history.mean_reward_per_round,
        "rounds": history.rounds,
        "initial_asr": history.initial_asr,
        "final_asr": history.final_asr,
    }


def _save_json(name, obj):
    with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _save_csv(name, rows, fields):
    with open(os.path.join(RESULTS_DIR, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_competitive_csv(competitive):
    rows = []
    for name, m in competitive.items():
        if not m.get("available"):
            rows.append({"defense": name, "available": False})
            continue
        rows.append({"defense": name, "available": True, "asr": m["asr"],
                     "asr_ci_lo": m["asr_ci"][0], "asr_ci_hi": m["asr_ci"][1],
                     "fpr": m["fpr"], "tpr": m["tpr"], "precision": m["precision"],
                     "f1": m["f1"], "ece": m.get("ece"),
                     "p95_ms": m["latency"]["p95"]})
    _save_csv("competitive.csv", rows,
              ["defense", "available", "asr", "asr_ci_lo", "asr_ci_hi", "fpr",
               "tpr", "precision", "f1", "ece", "p95_ms"])


def _print_leaderboard(competitive):
    print("\n" + "=" * 72)
    print("LEADERBOARD (sorted by Attack Success Rate, lower is better)")
    print("=" * 72)
    avail = [(n, m) for n, m in competitive.items() if m.get("available")]
    avail.sort(key=lambda x: x[1]["asr"])
    for name, m in avail:
        ci = m["asr_ci"]
        print(f"  {name:<42} ASR={m['asr']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
              f"FPR={m['fpr']:.3f}")
    for name, m in competitive.items():
        if not m.get("available"):
            print(f"  {name:<42} (skipped — unavailable)")
    print("=" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
