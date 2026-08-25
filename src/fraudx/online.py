"""Drift-aware dual-memory online classifier with delayed labels."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from river import compose, drift, linear_model, preprocessing


def _online_model() -> compose.Pipeline:
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(l2=1e-3),
    )


@dataclass
class DriftAwareDualMemory:
    """Blend fast and stable learners, increasing fast weight after drift."""

    long_update_stride: int = 5
    alpha_decay: float = 0.98
    drift_alpha: float = 0.85
    short_model: compose.Pipeline = field(default_factory=_online_model)
    long_model: compose.Pipeline = field(default_factory=_online_model)
    detector: drift.ADWIN = field(default_factory=drift.ADWIN)
    alpha: float = 0.5
    observations: int = 0
    drift_points: list[int] = field(default_factory=list)

    def predict_proba_one(self, features: dict[str, float]) -> float:
        short = self.short_model.predict_proba_one(features).get(1, 0.0)
        long = self.long_model.predict_proba_one(features).get(1, 0.0)
        return float(self.alpha * short + (1 - self.alpha) * long)

    def learn_one(
        self,
        features: dict[str, float],
        label: int,
        probability: float,
        false_negative_cost: float = 10.0,
    ) -> bool:
        prediction = int(probability >= 0.5)
        cost = float(prediction != label)
        if label == 1 and prediction == 0:
            cost *= false_negative_cost
        self.detector.update(cost)
        self.observations += 1
        detected = bool(self.detector.drift_detected)
        if detected:
            self.alpha = self.drift_alpha
            self.drift_points.append(self.observations)
        else:
            self.alpha = max(0.5, self.alpha * self.alpha_decay)
        self.short_model.learn_one(features, label)
        if self.observations % self.long_update_stride == 0:
            self.long_model.learn_one(features, label)
        return detected


def prequential_evaluate(
    rows: list[tuple[dict[str, float], int]],
    label_delay: int = 1,
    false_negative_cost: float = 10.0,
) -> dict[str, Any]:
    """Test-then-train stream evaluation with configurable delayed feedback."""
    if label_delay < 0:
        raise ValueError("Label delay cannot be negative")
    classifier = DriftAwareDualMemory()
    pending: deque[tuple[dict[str, float], int, float]] = deque()
    labels: list[int] = []
    probabilities: list[float] = []
    for features, label in rows:
        probability = classifier.predict_proba_one(features)
        labels.append(label)
        probabilities.append(probability)
        pending.append((features, label, probability))
        if len(pending) > label_delay:
            old_features, old_label, old_probability = pending.popleft()
            classifier.learn_one(
                old_features,
                old_label,
                old_probability,
                false_negative_cost=false_negative_cost,
            )
    while pending:
        old_features, old_label, old_probability = pending.popleft()
        classifier.learn_one(
            old_features,
            old_label,
            old_probability,
            false_negative_cost=false_negative_cost,
        )
    return {
        "labels": labels,
        "probabilities": probabilities,
        "drift_points": classifier.drift_points,
    }

