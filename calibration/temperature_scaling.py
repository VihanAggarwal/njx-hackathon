"""System 8 — temperature scaling.

Fit a single scalar temperature T on a held-out validation split by minimizing
negative log-likelihood: convert the score to a logit, divide by T, apply sigmoid,
and minimize NLL against the labels. Dividing a logit by T > 0 is a strictly
monotonic transform of the score, so it CANNOT change the ranking of scores — only
their probability values (and hence ECE) change.
"""

from __future__ import annotations

import numpy as np


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class TemperatureScaler:
    method = "temperature"

    def __init__(self):
        self.T = 1.0
        self.fitted = False

    def fit(self, scores, labels) -> "TemperatureScaler":
        z = _logit(scores)
        y = np.asarray(labels, dtype=float)

        def nll(T):
            T = max(float(T), 1e-3)
            p = 1.0 / (1.0 + np.exp(-z / T))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
            self.T = float(res.x)
        except Exception:
            # gradient-free fallback: coarse grid + local refine
            grid = np.linspace(0.1, 10.0, 100)
            self.T = float(grid[int(np.argmin([nll(t) for t in grid]))])
        self.fitted = True
        return self

    def calibrate(self, score):
        z = _logit(score) / max(self.T, 1e-3)
        p = 1.0 / (1.0 + np.exp(-z))
        return float(p) if np.ndim(score) == 0 else np.asarray(p)
