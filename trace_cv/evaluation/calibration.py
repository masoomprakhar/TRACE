"""Confidence calibration.

Raw softmax / detector scores are usually over-confident, so a reported "90%"
doesn't mean 90% empirical correctness. Temperature scaling fits a single
scalar T on a validation set to soften (or sharpen) scores so confidence
becomes trustworthy — important when a threshold routes borderline cases to
human review. Pure Python (no scipy/sklearn dependency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def expected_calibration_error(
    confidences: list[float], labels: list[int], n_bins: int = 10
) -> float:
    """ECE: weighted gap between confidence and accuracy across bins.
    `labels[i]` is 1 if prediction i was correct, else 0."""
    if not confidences:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, labels):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, y))
    n = len(confidences)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(avg_conf - acc)
    return ece


@dataclass
class TemperatureScaler:
    """1-D temperature scaling fitted by grid-search NLL minimization."""

    temperature: float = 1.0

    def fit(
        self,
        confidences: list[float],
        labels: list[int],
        t_min: float = 0.5,
        t_max: float = 10.0,
        step: float = 0.1,
    ) -> "TemperatureScaler":
        logits = [_logit(c) for c in confidences]
        best_t, best_nll = 1.0, float("inf")
        t = t_min
        while t <= t_max + 1e-9:
            nll = 0.0
            for z, y in zip(logits, labels):
                p = min(max(_sigmoid(z / t), 1e-6), 1.0 - 1e-6)
                nll -= y * math.log(p) + (1 - y) * math.log(1.0 - p)
            if nll < best_nll:
                best_nll, best_t = nll, t
            t += step
        self.temperature = round(best_t, 4)
        return self

    def transform(self, confidence: float) -> float:
        return _sigmoid(_logit(confidence) / self.temperature)

    def transform_many(self, confidences: list[float]) -> list[float]:
        return [self.transform(c) for c in confidences]

    def to_dict(self) -> dict:
        return {"temperature": self.temperature}
