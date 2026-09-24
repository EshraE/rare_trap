from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _scalarize_limit_state(value: Any) -> float:
    if isinstance(value, tuple):
        value = value[0]
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    flat = array.ravel()
    if flat.size != 1:
        raise ValueError(f"limit_state must be scalar-like, got shape {array.shape}.")
    return float(flat[0])


def _vectorize_limit_state(value: Any, expected_rows: int) -> np.ndarray:
    if isinstance(value, tuple):
        value = value[0]
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and int(array.shape[0]) == expected_rows:
        return array.astype(float, copy=False)
    if array.ndim == 2 and tuple(array.shape) == (expected_rows, 1):
        return array[:, 0].astype(float, copy=False)
    raise ValueError(
        f"limit_state batch output must have shape ({expected_rows},) or ({expected_rows}, 1), got {array.shape}."
    )


@dataclass(frozen=True)
class LimitStateTarget:
    dimension: int
    limit_state_fn: Callable[[np.ndarray], Any]

    def __post_init__(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError(f"dimension must be positive, got {self.dimension}.")

    def __call__(self, x: np.ndarray) -> dict[str, float | bool]:
        point = np.asarray(x, dtype=float).ravel()
        if point.size != int(self.dimension):
            raise ValueError(f"Expected point of length {self.dimension}, got {point.size}.")

        limit_state = _scalarize_limit_state(self.limit_state_fn(point))
        valid = math.isfinite(limit_state)
        return {
            "limit_state": limit_state,
            "log_target": 0.0,
            "valid": valid,
        }

    def evaluate_batch(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        points = np.asarray(x, dtype=float)
        if points.ndim != 2:
            raise ValueError(f"Expected a 2D batch of points, got shape {points.shape}.")
        if int(points.shape[1]) != int(self.dimension):
            raise ValueError(f"Expected points of width {self.dimension}, got {points.shape[1]}.")

        limit_state = _vectorize_limit_state(self.limit_state_fn(points), int(points.shape[0]))
        valid = np.isfinite(limit_state)
        values = np.where(valid, limit_state, np.inf).astype(float, copy=False)
        log_targets = np.where(valid, 0.0, -np.inf).astype(float, copy=False)
        return values, log_targets, int(np.count_nonzero(~valid))


def build_limit_state_target(
    dimension: int,
    limit_state_fn: Callable[[np.ndarray], Any],
) -> LimitStateTarget:
    return LimitStateTarget(
        dimension=int(dimension),
        limit_state_fn=limit_state_fn,
    )