"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Minimal strongly typed view of the experiment configuration."""

    seed: int
    data_path: Path
    processed_path: Path
    target: str
    time_column: str
    train_fraction: float
    validation_fraction: float
    windows: tuple[int, ...]
    output_dir: Path
    raw: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML and validate fractions and temporal windows."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = raw["data"]
    features = raw["features"]
    train_fraction = float(data["train_fraction"])
    validation_fraction = float(data["validation_fraction"])
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty temporal test interval is required")
    windows = tuple(sorted({int(value) for value in features["windows"]}))
    if not windows or windows[0] <= 0:
        raise ValueError("Temporal windows must contain positive integers")
    base = config_path.parent.parent
    return ExperimentConfig(
        seed=int(raw["seed"]),
        data_path=(base / data["path"]).resolve(),
        processed_path=(base / data["processed_path"]).resolve(),
        target=str(data["target"]),
        time_column=str(data["time_column"]),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        windows=windows,
        output_dir=(base / raw["output_dir"]).resolve(),
        raw=raw,
    )
