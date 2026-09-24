"""Estimator execution helpers for ThinkTrap objectives."""

from __future__ import annotations

from typing import Callable

import numpy as np

from estimators_samplers import (
    SubsetSimulationConfig,
    SubsetSimulationResult,
    SamplerTraceObserver,
    build_limit_state_target,
    subset_simulation as sampler_subset_simulation,
)
from utils.config import LEVEL_ZERO_SCREEN_PROBABILITY, LEVEL_ZERO_SCREEN_SAMPLES
from utils.reporting import ESTIMATOR_IMPLEMENTATION, print_level_progress, print_level_summaries
from utils.thinktrap_objective import ThinkTrapObjective


def run_subset(
    obj: ThinkTrapObjective,
    samples_per_level: int,
    conditional_prob: float,
    max_levels: int,
    seed: int,
    sampler_progress: bool,
    verbose: bool = True,
    trace_observer: SamplerTraceObserver | None = None,
    level_complete_callback: Callable[[object], None] | None = None,
) -> SubsetSimulationResult:
    def limit_fn(x: np.ndarray):
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            return obj.g_single(arr.astype(np.float32))
        if arr.ndim == 2:
            return obj(arr.astype(np.float32), to_numpy=True)
        raise ValueError(f"Expected 1D or 2D input, got shape {arr.shape}.")

    target = build_limit_state_target(dimension=obj.Dg, limit_state_fn=limit_fn)
    if verbose:
        print(
            f"[subset] estimator={ESTIMATOR_IMPLEMENTATION} | D={obj.Dg} | "
            f"N={int(samples_per_level)} | p0={float(conditional_prob):.4f} | "
            f"L={int(max_levels)} | "
            f"level0_screen=n{LEVEL_ZERO_SCREEN_SAMPLES},"
            f"p>={LEVEL_ZERO_SCREEN_PROBABILITY:.2f}"
        )
    def handle_level_complete(method_name: str, summary: object) -> None:
        if sampler_progress:
            print_level_progress(method_name, summary, obj)
        if level_complete_callback is not None:
            level_complete_callback(summary)

    result = sampler_subset_simulation(
        dimension=obj.Dg,
        target=target,
        config=SubsetSimulationConfig(
            samples_per_level=int(samples_per_level),
            conditional_prob=float(conditional_prob),
            max_levels=int(max_levels),
            random_seed=int(seed),
            progress=bool(sampler_progress or level_complete_callback is not None),
            progress_callback=handle_level_complete,
            evaluation_context_callback=obj.set_subset_level,
            trace_observer=trace_observer,
            level_zero_screen_samples=LEVEL_ZERO_SCREEN_SAMPLES,
            level_zero_screen_probability=LEVEL_ZERO_SCREEN_PROBABILITY,
        ),
    )
    if verbose:
        max_score = (
            float(obj.performance_threshold) - float(np.min(result.final_limit_state))
            if int(result.final_limit_state.size)
            else float("nan")
        )
        if str(obj.performance_metric) == "output_length":
            max_score_part = f"max_output_tokens={max_score:.1f}"
        else:
            max_score_part = f"max_repetition_score={max_score:.6f}"
        print()
        screen = result.level_zero_screen
        terminal_summary = result.level_summaries[-1]
        achieved_threshold = (
            float(obj.performance_threshold) - float(terminal_summary.threshold)
        )
        achieved_threshold_part = (
            f"achieved_output_token_threshold={achieved_threshold:.1f}"
            if str(obj.performance_metric) == "output_length"
            else f"achieved_repetition_score_threshold={achieved_threshold:.6f}"
        )
        target_probability_status = (
            "final" if result.reliable_probability else "nonfinal_diagnostic"
        )
        if screen is not None and screen.triggered:
            print(
                f"[subset] stop={result.stop_reason.value} "
                "probability_method=direct_monte_carlo_fixed_checkpoint "
                f"evals={result.total_evaluations} "
                f"successes={screen.observed_failures}/{screen.checkpoint_samples} "
                f"success_probability={result.estimated_probability:.6e} "
                f"ci95=[{screen.confidence_interval_lower:.6f},"
                f"{screen.confidence_interval_upper:.6f}] "
                f"target_reached={str(result.target_reached).lower()} "
                f"{achieved_threshold_part} "
                f"achieved_threshold_probability={result.terminal_threshold_probability:.6e} "
                f"target_probability_status={target_probability_status} "
                "conditional_subset_levels_skipped=true "
                f"reliable={result.reliable_probability} {max_score_part}"
            )
        else:
            print(
                f"[subset] stop={result.stop_reason.value} levels={result.completed_levels} "
                f"probability_method={result.probability_method} "
                f"evals={result.total_evaluations} "
                f"success_probability={result.estimated_probability:.6e} "
                f"target_probability_status={target_probability_status} "
                f"target_reached={str(result.target_reached).lower()} "
                f"{achieved_threshold_part} "
                f"achieved_threshold_probability={result.terminal_threshold_probability:.6e} "
                f"reliable={result.reliable_probability} "
                f"{max_score_part}"
            )
        print_level_summaries("subset", result.level_summaries, obj.threshold, obj=obj)
    return result
