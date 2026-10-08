"""Exact capacity, grouped-score AP and shared paired temporal resampling."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from .common import CELLS, FAMILIES, SEEDS, require, write_json

CAPACITIES = (0.01, 0.03, 0.05)
METRICS = ("ap",) + tuple(f"{metric}@{q:.2f}" for q in CAPACITIES for metric in ("tp", "recall", "precision"))
PRIMARY = {f"{family}/{contrast}/tp@0.03" for family in FAMILIES for contrast in ("H1", "D")}


def capacity(n: int, fraction: float) -> int:
    q = Fraction(str(fraction))
    require(n > 0 and 0 < q <= 1, "Capacity needs positive N and fraction in (0,1]")
    return (n * q.numerator + q.denominator - 1) // q.denominator


def priority(source_ids: np.ndarray) -> np.ndarray:
    return np.asarray([hashlib.sha256(f"banksim-capacity-v1|{int(i)}".encode("utf-8")).hexdigest() for i in source_ids])


class ScoreOrder:
    """Pre-sort once; integer multiplicities exactly equal literal row replication."""

    def __init__(self, labels: np.ndarray, scores: np.ndarray, source_ids: np.ndarray, priorities: np.ndarray | None = None) -> None:
        labels, scores, source_ids = np.asarray(labels), np.asarray(scores), np.asarray(source_ids)
        require(labels.ndim == scores.ndim == source_ids.ndim == 1, "Expected one-dimensional inputs")
        require(len(labels) == len(scores) == len(source_ids) > 0, "Metric input length mismatch")
        require(bool(np.isin(labels, [0, 1]).all()), "Labels must be binary")
        require(bool(np.isfinite(scores).all()), "Nonfinite scores")
        require(len(np.unique(source_ids)) == len(source_ids), "Original source identities must be unique")
        priorities = priority(source_ids) if priorities is None else priorities
        require(len(priorities) == len(scores), "Priority length mismatch")
        self.order = np.lexsort((source_ids, priorities, -scores))
        self.y = labels[self.order].astype(np.int64)
        self.scores = scores[self.order].astype(np.float64)
        self.ends = np.r_[np.flatnonzero(self.scores[1:] != self.scores[:-1]), len(scores) - 1]

    def evaluate(self, multiplicity: np.ndarray | None = None, fractions: tuple[float, ...] = CAPACITIES) -> dict[str, Any]:
        if multiplicity is None:
            weights = np.ones(len(self.y), dtype=np.int64)
        else:
            raw = np.asarray(multiplicity)
            require(raw.shape == self.y.shape, "Multiplicity alignment mismatch")
            require(bool(np.isfinite(raw).all() and (raw >= 0).all() and (raw == np.floor(raw)).all()), "Multiplicity must be nonnegative integers")
            weights = raw[self.order].astype(np.int64)
        counts = np.cumsum(weights, dtype=np.int64)
        positives = np.cumsum(weights * self.y, dtype=np.int64)
        n, n_pos = int(counts[-1]), int(positives[-1])
        require(n > 0, "Empty resample")
        ends = self.ends[counts[self.ends] > 0]
        if n_pos:
            tp = positives[ends]
            ap = float(np.sum(np.diff(np.r_[0, tp]) / n_pos * (tp / counts[ends])))
        else:
            ap = float("nan")
        if multiplicity is None and n_pos:
            # Point/tuning AP uses the declared sklearn function; weighted replay
            # uses its grouped-score formula, checked numerically against replication.
            ap = float(average_precision_score(self.y, self.scores))
        result: dict[str, Any] = {"n": n, "positives": n_pos, "ap": ap, "constant_scores": bool(self.scores[0] == self.scores[-1])}
        for q in fractions:
            k = capacity(n, q)
            boundary = int(np.searchsorted(counts, k, side="left"))
            count_before = int(counts[boundary - 1]) if boundary else 0
            tp_before = int(positives[boundary - 1]) if boundary else 0
            caught = tp_before + (k - count_before) * int(self.y[boundary])
            end_index = int(np.searchsorted(self.ends, boundary))
            upper = int(self.ends[end_index])
            lower = int(self.ends[end_index - 1]) + 1 if end_index else 0
            above_n = int(counts[lower - 1]) if lower else 0
            above_tp = int(positives[lower - 1]) if lower else 0
            tie_n = int(counts[upper]) - above_n
            tie_p = int(positives[upper]) - above_tp
            slots = k - above_n
            suffix = f"@{q:.2f}"
            result.update({"k" + suffix: k, "tp" + suffix: caught, "recall" + suffix: caught / n_pos if n_pos else float("nan"), "precision" + suffix: caught / k, "tie_n" + suffix: tie_n, "tie_positives" + suffix: tie_p, "tie_tp_min" + suffix: above_tp + max(0, slots - (tie_n - tie_p)), "tie_tp_max" + suffix: above_tp + min(slots, tie_p)})
        return result


def paired_summary(results: dict[str, dict[str, Any]], seeds: tuple[int, ...] = SEEDS) -> dict[str, float]:
    """Missing seeds propagate to missing cell/contrast; never select successful seeds."""
    summary: dict[str, float] = {}
    for family in FAMILIES:
        for metric in METRICS:
            cell_values: dict[str, np.ndarray] = {}
            for cell in CELLS:
                vals = np.asarray([results.get(f"{family}|{cell}|{seed}", {}).get(metric, float("nan")) for seed in seeds], dtype=float)
                cell_values[cell] = vals
                summary[f"{family}/{cell}/{metric}"] = float(np.mean(vals))
            h0 = cell_values["B0H"] - cell_values["B0"]
            h1 = cell_values["B1H"] - cell_values["B1"]
            for contrast, vals in (("H0", h0), ("H1", h1), ("D", h1 - h0)):
                summary[f"{family}/{contrast}/{metric}"] = float(np.mean(vals))
    for contrast in ("H0", "H1"):
        for metric in METRICS:
            summary[f"XGBoost-minus-LightGBM/{contrast}/{metric}"] = summary[f"XGBoost/{contrast}/{metric}"] - summary[f"LightGBM/{contrast}/{metric}"]
    return summary


def moving_blocks(time_steps: np.ndarray, length: int, draws: int, seed: int) -> np.ndarray:
    time_steps = np.asarray(time_steps, dtype=np.int64)
    require(time_steps.ndim == 1 and len(time_steps) > 0, "Time grid must be one-dimensional")
    require(bool(np.all(np.diff(time_steps) == 1)), "Moving blocks require consecutive steps")
    require(1 <= length <= len(time_steps) and draws > 0, "Invalid block configuration")
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.integers(0, len(time_steps) - length + 1, size=(draws, math.ceil(len(time_steps) / length)))
    indices = (starts[:, :, None] + np.arange(length)).reshape(draws, -1)[:, :len(time_steps)]
    return time_steps[indices]


def multiplicities(row_steps: np.ndarray, sampled_steps: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    lookup = np.searchsorted(time_grid, row_steps)
    require(bool((lookup < len(time_grid)).all()), "Rows outside resampling grid")
    require(bool(np.array_equal(time_grid[lookup], row_steps)), "Rows outside resampling grid")
    sampled_lookup = np.searchsorted(time_grid, sampled_steps)
    require(bool((sampled_lookup < len(time_grid)).all()), "Draw outside resampling grid")
    require(bool(np.array_equal(time_grid[sampled_lookup], sampled_steps)), "Draw outside resampling grid")
    return np.bincount(sampled_lookup, minlength=len(time_grid))[lookup]


def interval(values: np.ndarray, *, confidence: float = .95, min_valid: int = 4750) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    record: dict[str, Any] = {"draws": len(values), "valid": len(finite), "invalid": len(values) - len(finite), "confidence": confidence, "scope": "conditional_fixed_models_and_window", "method": "percentile_linear"}
    if len(finite) < min_valid or not len(finite):
        return {**record, "status": "insufficient_valid_draws", "lower": None, "upper": None}
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(finite, [alpha, 1 - alpha], method="linear")
    return {**record, "status": "reported", "lower": float(lo), "upper": float(hi)}


def bootstrap_orders(orders: dict[str, ScoreOrder], row_steps: np.ndarray, output: Path, *, draws: int = 5000, seeds: tuple[int, ...] = SEEDS, grid: np.ndarray | None = None, lengths: tuple[int, ...] = (7, 3, 14)) -> dict[str, Any]:
    """All score orders share original row alignment and each saved draw sequence."""
    output.mkdir(parents=True, exist_ok=True)
    time_grid = np.unique(row_steps) if grid is None else grid
    point = paired_summary({key: order.evaluate() for key, order in orders.items()}, seeds)
    names = list(point)
    write_json(output / "estimand_names.json", names)
    write_json(output / "point_estimates.json", point)
    intervals: list[dict[str, Any]] = []
    for length in lengths:
        seed = {7: 20260925, 3: 20260926, 14: 20260927}[length]
        sequences = moving_blocks(time_grid, length, draws, seed)
        np.save(output / f"step_sequences_L{length}.npy", sequences, allow_pickle=False)
        estimates = np.lib.format.open_memmap(output / f"draw_effects_L{length}.npy", mode="w+", dtype=np.float64, shape=(draws, len(names)))
        for i, seq in enumerate(sequences):
            weights = multiplicities(row_steps, seq, time_grid)
            summary = paired_summary({key: order.evaluate(weights) for key, order in orders.items()}, seeds)
            estimates[i] = [summary[name] for name in names]
            if (i + 1) % 100 == 0:
                estimates.flush()
                write_json(output / "progress.json", {"block_length": length, "draws_completed": i + 1, "planned_draws": draws})
        estimates.flush()
        for j, name in enumerate(names):
            row = {"estimand": name, "block_length": length, "estimate": point[name], **interval(estimates[:, j], min_valid=math.ceil(.95 * draws))}
            if name in PRIMARY:
                row["bonferroni_marginal"] = interval(estimates[:, j], confidence=.9875, min_valid=math.ceil(.95 * draws))
            intervals.append(row)
        del estimates
    record = {"status": "complete", "intervals": intervals, "no_equivalence_inference": True, "multiplicity_family": sorted(PRIMARY)}
    write_json(output / "intervals.json", record)
    return record
