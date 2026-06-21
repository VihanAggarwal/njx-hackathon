"""Tests for System 8 — the calibration layer.

Key guarantees: calibration is MONOTONIC (preserves score ranking, so ASR/FPR/F1
are unchanged), and it REDUCES ECE on overconfident inputs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import Calibrator, IsotonicCalibrator, TemperatureScaler
from eval import metrics as M


def _overconfident(n, seed):
    """Scores that are systematically overconfident vs the true probabilities."""
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p_true).astype(int)
    logit = np.log(p_true / (1 - p_true))
    s = 1.0 / (1.0 + np.exp(-3.0 * logit))   # sharpen -> overconfident
    return s, y


def test_temperature_monotonic_preserves_ranking():
    s, y = _overconfident(200, 1)
    ts = TemperatureScaler().fit(s, y)
    c = ts.calibrate(s)
    assert np.array_equal(np.argsort(s), np.argsort(c))


def test_temperature_reduces_ece_and_T_gt_1():
    s, y = _overconfident(400, 2)
    ts = TemperatureScaler().fit(s[:300], y[:300])
    before = M.expected_calibration_error(list(s[300:]), list(y[300:]))
    after = M.expected_calibration_error([ts.calibrate(x) for x in s[300:]], list(y[300:]))
    assert after < before
    assert ts.T > 1.0   # overconfident input -> temperature > 1


def test_threshold_partition_is_identical():
    # ASR/FPR depend only on which items fall on each side of a threshold. With a
    # monotonic map and the mapped threshold, that partition is IDENTICAL.
    s, y = _overconfident(300, 3)
    ts = TemperatureScaler().fit(s, y)
    t = 0.8
    raw_block = s > t
    cal_block = ts.calibrate(s) > ts.calibrate(t)
    assert np.array_equal(raw_block, cal_block)


def test_isotonic_monotonic_and_reduces_ece():
    s, y = _overconfident(400, 4)
    iso = IsotonicCalibrator().fit(s[:300], y[:300])
    c = np.array([iso.calibrate(x) for x in s])
    order = np.argsort(s)
    assert np.all(np.diff(c[order]) >= -1e-9)   # non-decreasing in s
    before = M.expected_calibration_error(list(s[300:]), list(y[300:]))
    after = M.expected_calibration_error([iso.calibrate(x) for x in s[300:]], list(y[300:]))
    assert after <= before + 1e-9


def test_calibrator_wrapper():
    s, y = _overconfident(200, 5)
    for m in ("temperature", "isotonic"):
        cal = Calibrator(m).fit(s, y)
        assert cal.fitted
        v = cal.calibrate(0.9)
        assert 0.0 <= v <= 1.0
    assert "method" in Calibrator("temperature").info
    with pytest.raises(ValueError):
        Calibrator("nope")
