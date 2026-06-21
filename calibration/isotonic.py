"""System 8 — isotonic regression calibrator (fallback method).

A non-parametric, non-decreasing mapping from score to calibrated probability fit
on the held-out validation split. Isotonic regression is monotonic by
construction, so it preserves score ranking (ties may merge, but order never
inverts). More flexible than temperature scaling but needs more data to avoid
overfitting — hence it's the fallback.
"""

from __future__ import annotations

import numpy as np


class IsotonicCalibrator:
    method = "isotonic"

    def __init__(self):
        self._iso = None
        self.fitted = False

    def fit(self, scores, labels) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
        self.fitted = True
        return self

    def calibrate(self, score):
        arr = np.atleast_1d(np.asarray(score, dtype=float))
        out = self._iso.predict(arr)
        return float(out[0]) if np.ndim(score) == 0 else np.asarray(out)
