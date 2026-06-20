"""Tests for eval/metrics.py — every metric checked on known synthetic inputs."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics as M


def _r(is_attack, blocked, score=0.5, cls="direct", lat=1.0):
    return {"is_attack": is_attack, "blocked": blocked, "score": score,
            "attack_class": cls, "latency_ms": lat}


# Known set: 4 attacks (3 blocked, 1 missed), 4 benign (1 blocked, 3 ok)
KNOWN = [
    _r(True, True), _r(True, True), _r(True, True), _r(True, False),   # TP=3 FN=1
    _r(False, False), _r(False, False), _r(False, False), _r(False, True),  # TN=3 FP=1
]


def test_confusion():
    c = M.confusion(KNOWN)
    assert c == {"TP": 3, "FP": 1, "TN": 3, "FN": 1}


def test_asr_fpr_tpr():
    assert M.attack_success_rate(KNOWN) == 0.25  # 1 of 4 attacks slipped through
    assert M.false_positive_rate(KNOWN) == 0.25  # 1 of 4 benign blocked
    assert M.true_positive_rate(KNOWN) == 0.75


def test_precision_recall_f1():
    # precision = 3/(3+1)=0.75, recall=0.75, f1=0.75
    assert M.precision(KNOWN) == 0.75
    assert abs(M.f1(KNOWN) - 0.75) < 1e-9


def test_per_class_asr():
    rs = [_r(True, True, cls="direct"), _r(True, False, cls="multi_hop_3"),
          _r(True, False, cls="multi_hop_3"), _r(False, False, cls="direct")]
    pc = M.per_class_asr(rs)
    assert pc["direct"] == 0.0          # the one direct attack was blocked
    assert pc["multi_hop_3"] == 1.0     # both multi-hop attacks slipped through


def test_percentile_and_latency():
    vals = [10, 20, 30, 40, 50]
    assert M.percentile(vals, 50) == 30
    assert M.percentile(vals, 0) == 10
    assert M.percentile(vals, 100) == 50
    rs = [_r(True, True, lat=v) for v in vals]
    ls = M.latency_stats(rs)
    assert ls["p50"] == 30


def test_bootstrap_ci_brackets_estimate_and_is_deterministic():
    lo, hi = M.bootstrap_asr_ci(KNOWN, n_resamples=500, seed=42)
    assert lo <= M.attack_success_rate(KNOWN) <= hi
    # deterministic with a fixed seed
    assert (lo, hi) == M.bootstrap_asr_ci(KNOWN, n_resamples=500, seed=42)


def test_ece_perfectly_calibrated_is_low():
    # scores equal observed frequency in each bin -> near-zero ECE
    scores = [0.05, 0.05, 0.95, 0.95]
    labels = [0, 0, 1, 1]
    assert M.expected_calibration_error(scores, labels, n_bins=10) < 0.1


def test_ece_miscalibrated_is_high():
    scores = [0.99, 0.99, 0.99, 0.99]   # very confident...
    labels = [0, 0, 0, 0]               # ...and always wrong
    assert M.expected_calibration_error(scores, labels, n_bins=10) > 0.9


def test_roc_auc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    roc = M.roc_points(scores, labels)
    assert roc["auc"] == 1.0
    pr = M.pr_points(scores, labels)
    assert pr["ap"] == 1.0


def test_summary_keys():
    s = M.summary(KNOWN)
    assert s["asr"] == 0.25 and s["fpr"] == 0.25
    assert s["n_attacks"] == 4 and s["n_benign"] == 4


def test_empty_inputs_safe():
    assert M.attack_success_rate([]) == 0.0
    assert M.false_positive_rate([]) == 0.0
    assert M.bootstrap_asr_ci([]) == (0.0, 0.0)
    assert M.expected_calibration_error([], []) == 0.0
