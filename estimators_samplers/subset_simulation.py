from __future__ import annotations

from typing import Optional

from .engine import (
    SubsetSimulationConfig,
    SubsetSimulationResult,
    TargetEvaluator,
    run_subset_simulation_engine,
)


def subset_simulation(
    dimension: int,
    target: TargetEvaluator,
    config: Optional[SubsetSimulationConfig] = None,
) -> SubsetSimulationResult:
    """Run subset simulation with independent defaults.

    Defaults are tuned for standard-Gaussian latent probability estimation:
    - standard initial population via ``initial_scale=1.0``
    - standard component-wise MH ratio
    - 4000 samples per level, ``p0=0.2``
    - stable sorting for deterministic handling of equal limit-state values
    """

    return run_subset_simulation_engine(dimension=dimension, target=target, config=config)
