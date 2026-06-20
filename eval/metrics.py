"""Eval — pure metric functions (no LLM dependencies).

Every function operates on plain Python lists of "result" records, each a dict:

    {
        "is_attack":   bool,    # ground truth
        "blocked":     bool,    # did the defense block it?
        "score":       float,   # confidence/risk in [0,1] (for ROC/PR/calibration)
        "attack_class": str,    # e.g. "direct", "indirect", "multi_hop_2", ...
        "latency_ms":  float,
    }

Definitions (attack-centric):
    TP = attack & blocked      FN = attack & not blocked
    FP = benign & blocked      TN = benign & not blocked
    ASR (Attack Success Rate) = FN / (#attacks)   -> lower is better
    FPR (False Positive Rate) = FP / (#benign)    -> lower is better
    TPR / recall              = TP / (#attacks)   = 1 - ASR
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Result = Dict


# --------------------------------------------------------------------------- #
# Core counts
# --------------------------------------------------------------------------- #
def confusion(results: Sequence[Result]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for r in results:
        atk, blk = bool(r["is_attack"]), bool(r["blocked"])
        if atk and blk:
            tp += 1
        elif atk and not blk:
            fn += 1
        elif not atk and blk:
            fp += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def attack_success_rate(results: Sequence[Result]) -> float:
    """Fraction of attacks that were NOT blocked. Lower is better."""
    atks = [r for r in results if r["is_attack"]]
    if not atks:
        return 0.0
    return _safe_div(sum(not r["blocked"] for r in atks), len(atks))


def false_positive_rate(results: Sequence[Result]) -> float:
    benign = [r for r in results if not r["is_attack"]]
    if not benign:
        return 0.0
    return _safe_div(sum(r["blocked"] for r in benign), len(benign))


def true_positive_rate(results: Sequence[Result]) -> float:
    return 1.0 - attack_success_rate(results) if any(r["is_attack"] for r in results) else 0.0


recall = true_positive_rate


def precision(results: Sequence[Result]) -> float:
    c = confusion(results)
    return _safe_div(c["TP"], c["TP"] + c["FP"])


def f1(results: Sequence[Result]) -> float:
    p, r = precision(results), recall(results)
    return _safe_div(2 * p * r, p + r)


def accuracy(results: Sequence[Result]) -> float:
    c = confusion(results)
    total = sum(c.values())
    return _safe_div(c["TP"] + c["TN"], total)


def per_class_asr(results: Sequence[Result]) -> Dict[str, float]:
    """ASR broken down by attack_class (benign records are ignored)."""
    classes: Dict[str, List[Result]] = {}
    for r in results:
        if r["is_attack"]:
            classes.setdefault(r.get("attack_class", "unknown"), []).append(r)
    return {cls: attack_success_rate(rs) for cls, rs in sorted(classes.items())}


def summary(results: Sequence[Result]) -> Dict[str, float]:
    return {
        "asr": round(attack_success_rate(results), 4),
        "fpr": round(false_positive_rate(results), 4),
        "tpr": round(true_positive_rate(results), 4),
        "precision": round(precision(results), 4),
        "f1": round(f1(results), 4),
        "accuracy": round(accuracy(results), 4),
        "n": len(results),
        "n_attacks": sum(r["is_attack"] for r in results),
        "n_benign": sum(not r["is_attack"] for r in results),
    }


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #
def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0,100])."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (p / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(xs[int(rank)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (rank - lo))


def latency_stats(results: Sequence[Result]) -> Dict[str, float]:
    lats = [r["latency_ms"] for r in results if "latency_ms" in r]
    return {
        "p50": round(percentile(lats, 50), 3),
        "p95": round(percentile(lats, 95), 3),
        "p99": round(percentile(lats, 99), 3),
        "mean": round(sum(lats) / len(lats), 3) if lats else 0.0,
    }


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    results: Sequence[Result],
    statistic: Callable[[Sequence[Result]], float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for a statistic over result records."""
    results = list(results)
    if not results:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(results)
    stats = []
    for _ in range(n_resamples):
        sample = [results[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[min(n_resamples - 1, max(0, int((alpha / 2) * n_resamples)))]
    hi = stats[min(n_resamples - 1, max(0, int((1 - alpha / 2) * n_resamples)))]
    return (round(lo, 4), round(hi, 4))


def bootstrap_asr_ci(results, n_resamples=1000, seed=42):
    return bootstrap_ci(results, attack_success_rate, n_resamples, seed=seed)


def bootstrap_fpr_ci(results, n_resamples=1000, seed=42):
    return bootstrap_ci(results, false_positive_rate, n_resamples, seed=seed)


# --------------------------------------------------------------------------- #
# Calibration (ECE + reliability diagram)
# --------------------------------------------------------------------------- #
def expected_calibration_error(
    scores: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """ECE: weighted gap between predicted confidence and observed accuracy."""
    if not scores:
        return 0.0
    n = len(scores)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores)
               if (s > lo or (b == 0 and s >= lo)) and s <= hi]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
    return round(ece, 4)


def reliability_curve(
    scores: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> Dict[str, List[float]]:
    """Per-bin (mean predicted prob, observed frequency, count) for plotting."""
    conf, acc, counts = [], [], []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores)
               if (s > lo or (b == 0 and s >= lo)) and s <= hi]
        if not idx:
            conf.append((lo + hi) / 2); acc.append(0.0); counts.append(0)
            continue
        conf.append(sum(scores[i] for i in idx) / len(idx))
        acc.append(sum(labels[i] for i in idx) / len(idx))
        counts.append(len(idx))
    return {"confidence": conf, "accuracy": acc, "counts": counts}


# --------------------------------------------------------------------------- #
# ROC / PR curves + AUC
# --------------------------------------------------------------------------- #
def roc_points(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, List[float]]:
    """ROC curve points + AUC (label 1 = attack)."""
    try:
        from sklearn.metrics import roc_curve, auc
        if len(set(labels)) < 2:
            return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}
        fpr, tpr, _ = roc_curve(labels, scores)
        return {"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr)),
                "auc": round(float(auc(fpr, tpr)), 4)}
    except Exception:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "auc": 0.5}


def pr_points(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, List[float]]:
    """Precision-Recall curve points + average precision."""
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        if len(set(labels)) < 2:
            return {"precision": [1.0, 0.0], "recall": [0.0, 1.0], "ap": 0.0}
        prec, rec, _ = precision_recall_curve(labels, scores)
        return {"precision": list(map(float, prec)), "recall": list(map(float, rec)),
                "ap": round(float(average_precision_score(labels, scores)), 4)}
    except Exception:
        return {"precision": [1.0, 0.0], "recall": [0.0, 1.0], "ap": 0.0}
