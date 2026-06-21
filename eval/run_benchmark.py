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
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_SCORES_FILE = os.path.join(RESULTS_DIR, "_hf_scores_protectai.json")
PG_SCORES_FILE = os.path.join(RESULTS_DIR, "_hf_scores_promptguard.json")
USER_GOAL = "Summarize this content for me."

# Real HF models scored torch-free from ungated ONNX weights in the clean venv.
# Each spec -> one sidecar the matching baseline reads. Prompt-Guard's official repo
# is gated, so we use an ungated ONNX mirror of the same weights (the workaround).
_HF_ONNX_MODELS = [
    {"label": "ProtectAI deberta-v3", "out": HF_SCORES_FILE,
     "model": "protectai/deberta-v3-base-prompt-injection-v2",
     "subfolder": "onnx", "file_name": "model.onnx"},
    {"label": "Meta Prompt-Guard-2 86M", "out": PG_SCORES_FILE,
     "model": "gravitee-io/Llama-Prompt-Guard-2-86M-onnx",
     "subfolder": "root", "file_name": "model.quant.onnx"},
]


def _prepare_hf_scores(items, cfg):
    """Score eval items with the REAL HF models (ProtectAI + Prompt-Guard) in a
    clean venv, torch-free via ONNX. Writes one sha256(content)->p_injection sidecar
    per model so those baselines report real numbers in the main venv. No-op for a
    model (baseline stays SKIPPED, never fabricated) if its scoring can't run.
    """
    import subprocess
    py = (cfg.get("eval", {}).get("onnx_python")
          or os.path.join(REPO_ROOT, ".venv-ml", "Scripts", "python.exe"))
    runner = os.path.join(REPO_ROOT, "eval", "baselines", "hf_onnx_runner.py")
    if not os.path.exists(py):
        print(f"  (no clean ONNX venv at {py}; HF model baselines -> SKIPPED)")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    items_path = os.path.join(RESULTS_DIR, "_hf_items.json")
    json.dump([{"content": it["content"]} for it in items],
              open(items_path, "w", encoding="utf-8"))
    env = dict(os.environ)
    env.setdefault("SSL_CERT_FILE",
                   os.path.join(os.path.expanduser("~"), ".local", "cacert.pem"))
    env.setdefault("REQUESTS_CA_BUNDLE", env["SSL_CERT_FILE"])
    for spec in _HF_ONNX_MODELS:
        print(f"  scoring items with REAL {spec['label']} ONNX model (clean venv)...")
        cmd = [py, runner, "--model", spec["model"], "--subfolder", spec["subfolder"],
               "--file-name", spec["file_name"], "--in", items_path, "--out", spec["out"]]
        for attempt in (1, 2):  # one retry: ONNX load can transiently OOM
            try:
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
                tail = (r.stdout.strip().splitlines() or ["(no stdout)"])[-1]
                print("   " + tail)
                if r.returncode == 0:
                    break
                err = (r.stderr.strip().splitlines() or [""])[-1]
                note = "retrying" if attempt == 1 else "SKIPPED"
                print(f"   {spec['label']} scoring failed -> {note}: {err}")
            except Exception as e:  # pragma: no cover - subprocess/env dependent
                note = "retrying" if attempt == 1 else "SKIPPED"
                print(f"   {spec['label']} scoring error -> {note}: {e}")


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
        DualMindConfig(True, True, True, True, "7_full_calibrated", True),
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


def evaluate_config(flags, provider, prefilter, config, items, calibrator=None):
    dm = DualMind(config, provider=provider,
                  prefilter=prefilter if flags.use_prefilter else None, flags=flags,
                  calibrator=calibrator)
    cost_before = getattr(provider, "cost", None)
    c0 = cost_before.uncached_cost_usd if cost_before else 0.0
    results = []
    for it in items:
        tr = dm.process(it["content"], user_goal=USER_GOAL,
                        content_type=it.get("content_type", "text"),
                        ground_truth="attack" if it["is_attack"] else "benign")
        # For attacks "blocked" means DEFENDED (harm prevented, incl. Decider
        # refusal); for benign it means over-blocked (a false positive).
        blocked = tr.defended if it["is_attack"] else (tr.final_verdict == "block")
        results.append({
            "is_attack": it["is_attack"], "blocked": blocked, "score": tr.risk,
            "calib_score": tr.calib_score,
            "attack_class": it["attack_class"], "latency_ms": tr.latency_ms,
            "dataset": it.get("dataset", ""), "caught_by": tr.caught_by,
            "path": tr.path, "llm_calls": tr.llm_calls,
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
            "calib_score": r.score,  # a baseline's score is its own confidence
            "attack_class": it["attack_class"], "latency_ms": r.latency_ms,
            "dataset": it.get("dataset", ""),
        })
    return results


def metrics_block(results, n_resamples, seed, cost_per_req=0.0):
    scores = [r["score"] for r in results]
    # ECE/reliability use the calibrated confidence (System 8), not the raw risk.
    cal = [r.get("calib_score", r["score"]) for r in results]
    labels = [1 if r["is_attack"] else 0 for r in results]
    s = M.summary(results)
    s.update({
        "asr_ci": M.bootstrap_asr_ci(results, n_resamples, seed=seed),
        "fpr_ci": M.bootstrap_fpr_ci(results, n_resamples, seed=seed),
        "per_class_asr": M.per_class_asr(results),
        "latency": M.latency_stats(results),
        "confusion": M.confusion(results),
        "ece": M.expected_calibration_error(cal, labels),
        "reliability": M.reliability_curve(cal, labels),
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

    # Harden on a red-team pool that is PROVABLY DISJOINT from the eval set. The
    # synthetic template space is small, so a different seed alone is NOT enough
    # (strings overlap verbatim); we dedup by content against the eval items and
    # draw across many seeds until we have enough non-overlapping attacks.
    from eval.datasets import synthetic_generator as synth
    eval_contents = {it["content"] for it in items}
    redteam_pool, rt_seed = [], seed + 777
    while len(redteam_pool) < 24 and rt_seed < seed + 877:
        for a in synth.generate_attacks(8, seed=rt_seed):
            c = a["content"]
            if c not in eval_contents and c not in redteam_pool:
                redteam_pool.append(c)
        rt_seed += 1
    leak = sum(1 for c in redteam_pool if c in eval_contents)
    assert leak == 0, f"redteam/eval leakage: {leak} overlapping items"
    print(f"  red-team pool: {len(redteam_pool)} attacks, disjoint from eval "
          f"(leakage check: {leak} overlaps)")
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

    # --- System 8: fit the calibration layer on a HELD-OUT split -------- #
    from calibration import Calibrator
    cal_cfg = cfg.get("calibration", {})
    cal_method = cal_cfg.get("method", "temperature")
    holdout_frac = cal_cfg.get("holdout_frac", 0.15)
    # A dedicated held-out set, sized for a stable fit (a tiny 15% of a small eval
    # set leaves too few points and temperature collapses to 1.0).
    n_cal = max(cal_cfg.get("min_holdout", 48), int(round(holdout_frac * len(items))))
    seen_all = set(eval_contents) | set(redteam_pool)
    n_each = max(1, n_cal // 2)            # balance attacks vs benign for a stable fit
    atk_pool, ben_pool, cs = [], [], seed + 555
    while (len(atk_pool) < n_each or len(ben_pool) < n_each) and cs < seed + 905:
        for a in synth.generate_attacks(4, seed=cs):
            if a["content"] not in seen_all and len(atk_pool) < n_each:
                seen_all.add(a["content"]); atk_pool.append(a)
        for b in synth.generate_benign(8, seed=cs + 10000):
            if b["content"] not in seen_all and len(ben_pool) < n_each:
                seen_all.add(b["content"]); ben_pool.append(b)
        cs += 1
    cal_items = atk_pool + ben_pool
    dm_fit = DualMind(cfg, provider=provider, prefilter=hardened_prefilter,
                      flags=DualMindConfig(True, True, True, True, "_cal_fit"))
    cal_scores, cal_labels = [], []
    for it in cal_items:
        tr = dm_fit.process(it["content"], user_goal=USER_GOAL,
                            content_type=it.get("content_type", "text"),
                            ground_truth="attack" if it["is_attack"] else "benign")
        cal_scores.append(tr.risk); cal_labels.append(1 if it["is_attack"] else 0)
    calibrator = Calibrator(cal_method)
    if len(set(cal_labels)) == 2:
        calibrator.fit(cal_scores, cal_labels)
    n_atk = sum(cal_labels)
    print("\n### CALIBRATION LAYER (System 8) ###")
    print(f"  method={cal_method}  held-out split: {len(cal_items)} items "
          f"({n_atk} attacks / {len(cal_labels) - n_atk} benign), "
          f"disjoint from eval + red-team pool")
    print(f"  fitted: {calibrator.info}")

    # --- run the ablation matrix --------------------------------------- #
    print("\n### ABLATION MATRIX ###")
    configs = {}
    per_attack_rows = []
    for flags in ablation_flags():
        pf = hardened_prefilter if flags.name.startswith(("6_", "7_")) else base_prefilter
        cal = calibrator if flags.use_calibration else None
        results, cost = evaluate_config(flags, provider, pf, cfg, items, calibrator=cal)
        configs[flags.name] = metrics_block(results, n_resamples, seed, cost)
        for r in results:
            per_attack_rows.append({"config": flags.name, **r})
        m = configs[flags.name]
        print(f"  {flags.name:<24} ASR={m['asr']:.3f} {tuple(m['asr_ci'])}  "
              f"FPR={m['fpr']:.3f}  F1={m['f1']:.3f}  ECE={m['ece']:.3f}  "
              f"p95={m['latency']['p95']}ms")

    # --- calibration effect: confirm metrics unchanged, ECE improved ---- #
    pre, post = configs.get("6_full_post_hardening"), configs.get("7_full_calibrated")
    if pre and post:
        unchanged = {k: (abs(pre[k] - post[k]) < 1e-9) for k in
                     ("asr", "fpr", "tpr", "precision", "f1")}
        all_same = all(unchanged.values())
        print("\n### CALIBRATION EFFECT (System 8) ###")
        print(f"  security metrics unchanged by calibration: {all_same}  {unchanged}")
        print(f"    ASR {pre['asr']} -> {post['asr']} | FPR {pre['fpr']} -> {post['fpr']} "
              f"| F1 {pre['f1']} -> {post['f1']}")
        print(f"  ECE {pre['ece']:.4f} (uncalibrated)  ->  {post['ece']:.4f} (calibrated)  "
              f"[{'improved' if post['ece'] < pre['ece'] else 'no change'}]")

    # --- latency profile: REAL per-stage cost + fast-path fraction ------ #
    from perf.latency_profiler import (measure_stage_latencies, synthesize_latencies,
                                       percentiles, fast_path_fraction)
    full_sample = [it for it in items if it["is_attack"]
                   and str(it.get("attack_class", "")).startswith("multi_hop")][:3]
    if len(full_sample) < 3:
        full_sample += [it for it in items if it["is_attack"]][: 3 - len(full_sample)]
    fast_sample = [it for it in items if not it["is_attack"]][:2]
    stage = measure_stage_latencies(cfg, hardened_prefilter, calibrator,
                                    full_sample, fast_sample, USER_GOAL)
    paths_by_cfg = {}
    for row in per_attack_rows:
        paths_by_cfg.setdefault(row["config"], []).append(row.get("path", "full_dual_llm"))
    # SAFETY: in a FULL config (pre-filter + dual-LLM), the fast-allow early-exit
    # must never skip an actual attack the dual-LLM would have caught, or ASR would
    # silently drop. (In no-defense / pre-filter-only there is no dual-LLM to skip,
    # so a "fast_allow" path there is expected and not a safety concern.)
    full_names = {f.name for f in ablation_flags() if f.use_prefilter and f.use_dual_llm}
    atk_fast = sum(1 for r in per_attack_rows if r.get("is_attack")
                   and r.get("path") == "fast_allow" and r["config"] in full_names)
    assert atk_fast == 0, (f"{atk_fast} attack(s) fast-allowed in a full config — lower "
                           "perf.fast_allow_threshold (it's above an attack's prefilter score)")
    print("\n### LATENCY PROFILE (real per-stage cost, cache disabled) ###")
    print(f"  early-exit safety: attacks fast-allowed in full configs = {atk_fast} (must be 0)")
    print(f"  median stage cost: prefilter={stage['prefilter_ms']:.1f}ms  "
          f"reader={stage['reader_ms']:.0f}ms  decider={stage['decider_ms']:.0f}ms")
    for name in (f.name for f in ablation_flags()):
        paths = paths_by_cfg.get(name, [])
        if not paths:
            continue
        lat = synthesize_latencies(paths, stage)
        pc = percentiles(lat); ff = fast_path_fraction(paths)
        dist = {p: paths.count(p) for p in sorted(set(paths))}
        configs[name]["latency_real"] = pc
        configs[name]["fast_path_fraction"] = ff
        configs[name]["path_dist"] = dist
        configs[name]["stage_ms"] = stage
        print(f"  {name:<22} fast-path={ff * 100:4.0f}%  real p50={pc['p50']:.0f}ms "
              f"p95={pc['p95']:.0f}ms p99={pc['p99']:.0f}ms  paths={dist}")

    # --- baselines ------------------------------------------------------ #
    competitive = {}
    if not args.no_baselines:
        print("\n### COMPETITIVE BASELINES ###")
        _prepare_hf_scores(items, cfg)
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

    # Add the FULL DUALMIND (post-hardening + System 8 calibration) to the
    # competitive set. Calibration is monotonic, so ASR/FPR/precision/recall/F1 are
    # identical to config 6 — but the reported probability (hence ECE) is the
    # calibrated one, which is the fair number for a head-to-head.
    competitive["DUALMIND (full)"] = {"available": True,
                                      **configs["7_full_calibrated"]}

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
        "redteam_pool_size": len(redteam_pool),
        "redteam_eval_leakage": 0,  # asserted disjoint above
        "calibration": {**calibrator.info, "holdout_items": len(cal_items),
                        "holdout_frac": holdout_frac},
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
