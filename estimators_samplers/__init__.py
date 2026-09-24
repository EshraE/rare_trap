"""Subset simulation for a standard-Gaussian latent reference law.

This package exposes subset simulation with usable defaults:

    from estimators_samplers import subset_simulation

The target callable must accept a single ``numpy.ndarray`` point and return
either:

    {"limit_state": g_value, "log_target": log_target_value}

or an ``Evaluation`` instance.

The engine applies the standard-normal Metropolis ratio directly. ``log_target``
is retained for interface compatibility and diagnostics, not used to define a
different reference density.
"""

from .engine import (
    Evaluation,
    LevelSummary,
    LevelZeroMonteCarloScreen,
    SamplerStopReason,
    SamplerTraceObserver,
    SubsetSimulationConfig,
    SubsetSimulationResult,
    TargetEvaluator,
)
from .trace_observer import SubsetTraceObserver
from .subset_simulation import subset_simulation
from .targets import LimitStateTarget, build_limit_state_target

__all__ = [
    "Evaluation",
    "TargetEvaluator",
    "LimitStateTarget",
    "LevelSummary",
    "LevelZeroMonteCarloScreen",
    "SamplerStopReason",
    "SamplerTraceObserver",
    "SubsetTraceObserver",
    "SubsetSimulationConfig",
    "SubsetSimulationResult",
    "build_limit_state_target",
    "subset_simulation",
]
