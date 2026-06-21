"""System 8 — calibration wrapper.

Exposes a uniform `.fit(scores, labels)` / `.calibrate(score)` interface over the
two monotonic calibrators. Default = temperature scaling; switch to isotonic via
config (`calibration.method: isotonic`).

This layer sits AFTER the aggregate risk score is computed and is used to attach a
calibrated probability for reporting / the review-gate's probabilistic thresholds.
Because every method here is monotonic, it never changes score ranking — so ASR,
FPR, precision, recall, and F1 are unaffected; only the probabilities and ECE move.
"""

from __future__ import annotations

from .isotonic import IsotonicCalibrator
from .temperature_scaling import TemperatureScaler


class Calibrator:
    def __init__(self, method: str = "temperature"):
        self.method = method
        if method == "isotonic":
            self._impl = IsotonicCalibrator()
        elif method == "temperature":
            self._impl = TemperatureScaler()
        else:
            raise ValueError(f"unknown calibration method: {method}")

    @property
    def fitted(self) -> bool:
        return self._impl.fitted

    def fit(self, scores, labels) -> "Calibrator":
        self._impl.fit(scores, labels)
        return self

    def calibrate(self, score):
        return self._impl.calibrate(score)

    @property
    def info(self) -> dict:
        d = {"method": self.method}
        if hasattr(self._impl, "T"):
            d["temperature"] = round(self._impl.T, 4)
        return d
