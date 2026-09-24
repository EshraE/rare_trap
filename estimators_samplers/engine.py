"""Subset simulation for a standard-Gaussian latent reference distribution.

The componentwise Metropolis kernel implements the Gaussian prior ratio directly.
Target ``log_target`` values are retained for interface compatibility and diagnostics;
they do not define the transition density in this engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol

import numpy as np


_LOG_FLOOR = 1e-300


class SamplerStopReason(str, Enum):
    CONVERGED = "converged"
    DIRECT_MONTE_CARLO_SCREEN = "direct_monte_carlo_screen"
    MAX_LEVELS = "max_levels"
    LOW_ACCEPTANCE = "low_acceptance"
    INVALID_EVALUATIONS = "invalid_evaluations"


@dataclass(frozen=True)
class Evaluation:
    limit_state: float
    log_target: float
    valid: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class TargetEvaluator(Protocol):
    def __call__(self, x: np.ndarray) -> Any:
        ...


class SamplerTraceObserver(Protocol):
    def observe_population(self, **kwargs: Any) -> None:
        ...

    def observe_transition_batch(self, **kwargs: Any) -> None:
        ...


@dataclass(frozen=True)
class LevelSummary:
    level: int
    threshold: float
    min_g: float
    mean_g: float
    max_g: float
    n_fail: int
    n_le_threshold: int
    population_size: int
    population_unique_points: int
    evaluations: int
    elite_count: int
    chain_length: int
    proposal_half_width: float
    invalid_evaluations: int = 0
    proposed_moves: int = 0
    accepted_moves: int = 0
    indicator_acceptance_rate: Optional[float] = None
    probability_mass_before_level: float = 1.0
    threshold_exceedance_probability: float = 0.0
    target_exceedance_probability: float = 0.0
    target_reached: bool = False


@dataclass(frozen=True)
class LevelZeroMonteCarloScreen:
    checkpoint_samples: int
    probability_boundary: float
    required_failures: int
    observed_failures: int
    estimated_probability: float
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    triggered: bool


@dataclass
class SubsetSimulationConfig:
    samples_per_level: int = 4000
    conditional_prob: float = 0.2
    max_levels: int = 20
    proposal_half_width: float = 1.0
    adapt_proposal_half_width: bool = True
    target_indicator_acceptance: float = 0.3
    adaptation_gain: float = 5.0
    max_width_change_factor: float = 3.0
    min_proposal_half_width: float = 0.1
    max_proposal_half_width: float = 1.0
    initial_scale: float = 1.0
    failure_threshold: float = 0.0
    random_seed: Optional[int] = None
    min_indicator_acceptance: float = 0.02
    progress: bool = True
    progress_callback: Optional[Callable[[str, LevelSummary], None]] = None
    evaluation_context_callback: Optional[Callable[[int], None]] = None
    trace_observer: Optional[SamplerTraceObserver] = None
    level_zero_screen_samples: Optional[int] = None
    level_zero_screen_probability: float = 0.1


@dataclass(frozen=True)
class SubsetSimulationResult:
    stop_reason: SamplerStopReason
    total_evaluations: int
    completed_levels: int
    level_summaries: list[LevelSummary]
    final_population: np.ndarray
    final_limit_state: np.ndarray
    final_log_target: np.ndarray
    level_probabilities: np.ndarray
    estimated_probability: float
    reliable_probability: bool
    level_zero_screen: Optional[LevelZeroMonteCarloScreen] = None
    probability_method: str = "subset_simulation"
    target_reached: bool = False
    terminal_threshold: float = float("nan")
    terminal_threshold_probability: float = 0.0


@dataclass
class _EngineState:
    points: np.ndarray
    values: np.ndarray
    log_targets: np.ndarray
    total_evaluations: int
    level_summaries: list[LevelSummary]
    completed_levels: int = 0
    level_probabilities: list[float] = field(default_factory=list)


def _count_unique_population_points(points: np.ndarray) -> int:
    population = np.asarray(points, dtype=float)
    if population.ndim != 2:
        raise ValueError(f"Expected a 2D population, got shape {population.shape}.")
    if int(population.shape[0]) == 0:
        return 0
    return int(np.unique(population, axis=0).shape[0])


def _normalize_scalar(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    flat = array.ravel()
    if flat.size != 1:
        raise ValueError(f"{name} must be scalar-like, got shape {array.shape}.")
    return float(flat[0])


def _normalize_evaluation(output: Any) -> Evaluation:
    if isinstance(output, Evaluation):
        result = output
    elif isinstance(output, Mapping):
        if "limit_state" not in output or "log_target" not in output:
            raise KeyError("Target output must include 'limit_state' and 'log_target'.")
        result = Evaluation(
            limit_state=_normalize_scalar(output["limit_state"], "limit_state"),
            log_target=_normalize_scalar(output["log_target"], "log_target"),
            valid=bool(output.get("valid", True)),
            details={
                key: value
                for key, value in output.items()
                if key not in {"limit_state", "log_target", "valid"}
            },
        )
    elif isinstance(output, tuple) and len(output) == 2:
        result = Evaluation(
            limit_state=_normalize_scalar(output[0], "limit_state"),
            log_target=_normalize_scalar(output[1], "log_target"),
        )
    else:
        raise TypeError(
            "Target output must be an Evaluation, a mapping with limit_state/log_target, "
            "or a two-tuple of (limit_state, log_target)."
        )

    if not math.isfinite(result.limit_state) or not math.isfinite(result.log_target):
        return Evaluation(
            limit_state=float("inf"),
            log_target=float("-inf"),
            valid=False,
            details={**result.details, "invalid_reason": "non_finite_output"},
        )
    return result


def _validate_common_config(
    samples_per_level: int,
    conditional_prob: float,
    max_levels: int,
    proposal_half_width: float,
    target_indicator_acceptance: float,
    adaptation_gain: float,
    max_width_change_factor: float,
    min_proposal_half_width: float,
    max_proposal_half_width: float,
    initial_scale: float,
    min_indicator_acceptance: float,
    level_zero_screen_samples: Optional[int],
    level_zero_screen_probability: float,
) -> None:
    if samples_per_level < 4:
        raise ValueError(f"samples_per_level must be >= 4, got {samples_per_level}.")
    if not (0.0 < conditional_prob < 1.0):
        raise ValueError(f"conditional_prob must be in (0, 1), got {conditional_prob}.")
    if max_levels < 0:
        raise ValueError(f"max_levels must be >= 0, got {max_levels}.")
    if proposal_half_width <= 0.0:
        raise ValueError(f"proposal_half_width must be positive, got {proposal_half_width}.")
    if not (0.0 < target_indicator_acceptance < 1.0):
        raise ValueError(
            "target_indicator_acceptance must be in (0, 1), got "
            f"{target_indicator_acceptance}."
        )
    if adaptation_gain <= 0.0:
        raise ValueError(f"adaptation_gain must be positive, got {adaptation_gain}.")
    if max_width_change_factor < 1.0:
        raise ValueError(
            "max_width_change_factor must be >= 1.0, got "
            f"{max_width_change_factor}."
        )
    if min_proposal_half_width <= 0.0:
        raise ValueError(
            "min_proposal_half_width must be positive, got "
            f"{min_proposal_half_width}."
        )
    if max_proposal_half_width < min_proposal_half_width:
        raise ValueError(
            "max_proposal_half_width must be >= min_proposal_half_width, got "
            f"[{min_proposal_half_width}, {max_proposal_half_width}]."
        )
    if initial_scale <= 0.0:
        raise ValueError(f"initial_scale must be positive, got {initial_scale}.")
    if min_indicator_acceptance < 0.0:
        raise ValueError(
            f"min_indicator_acceptance must be >= 0, got {min_indicator_acceptance}."
        )
    if level_zero_screen_samples is not None and level_zero_screen_samples < 1:
        raise ValueError(
            "level_zero_screen_samples must be positive when configured, got "
            f"{level_zero_screen_samples}."
        )
    if not (0.0 < level_zero_screen_probability < 1.0):
        raise ValueError(
            "level_zero_screen_probability must be in (0, 1), got "
            f"{level_zero_screen_probability}."
        )
    elite_count = int(round(samples_per_level * conditional_prob))
    elite_count = max(1, min(elite_count, samples_per_level - 1))
    if elite_count < 2:
        raise ValueError(
            "Configuration yields fewer than 2 elite seeds per level. Increase samples_per_level "
            "or conditional_prob."
        )


def _elite_count(samples_per_level: int, conditional_prob: float) -> int:
    elite_count = int(round(samples_per_level * conditional_prob))
    return max(1, min(elite_count, samples_per_level - 1))


def _chain_length(samples_per_level: int, elite_count: int) -> int:
    return max(1, int(math.ceil((samples_per_level - elite_count) / elite_count)))


def _wilson_score_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}.")
    if successes < 0 or successes > trials:
        raise ValueError(f"successes must be in [0, trials], got {successes}/{trials}.")

    probability = float(successes) / float(trials)
    z_squared = z * z
    denominator = 1.0 + z_squared / float(trials)
    center = (probability + z_squared / (2.0 * float(trials))) / denominator
    half_width = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / float(trials)
            + z_squared / (4.0 * float(trials) * float(trials))
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _print_run_config(
    method_name: str,
    dimension: int,
    samples_per_level: int,
    conditional_prob: float,
    max_levels: int,
    proposal_half_width: float,
    adapt_proposal_half_width: bool,
    target_indicator_acceptance: float,
) -> None:
    parts = [
        f"[{method_name}] D={dimension}",
        f"N={samples_per_level}",
        f"p0={conditional_prob:.4f}",
        f"L={max_levels}",
        f"w0={proposal_half_width:.4f}",
    ]
    if adapt_proposal_half_width:
        parts.append(f"adapt={target_indicator_acceptance:.2f}")
    else:
        parts.append("adapt=off")
    print(" | ".join(parts), flush=True)


def _print_level_progress(method_name: str, summary: LevelSummary) -> None:
    parts = [f"[{method_name}] level={summary.level:02d}"]
    if summary.level == 0:
        parts.extend(
            [
                f"min_g={summary.min_g:.6e}",
                f"mean_g={summary.mean_g:.6e}",
                f"max_g={summary.max_g:.6e}",
                f"evals={summary.evaluations}",
            ]
        )
    else:
        parts.extend(
            [
                f"thr={summary.threshold:.6e}",
                f"min_g={summary.min_g:.6e}",
                f"mean_g={summary.mean_g:.6e}",
                f"max_g={summary.max_g:.6e}",
                f"evals={summary.evaluations}",
                f"w={summary.proposal_half_width:.4f}",
            ]
        )
    if summary.proposed_moves > 0:
        accept_rate = summary.accepted_moves / max(summary.proposed_moves, 1)
        parts.append(f"accept={accept_rate:.4f}")
    parts.append(
        f"threshold_probability={summary.threshold_exceedance_probability:.6e}"
    )
    parts.append(
        "threshold_probability_basis="
        + (
            "fixed_checkpoint_empirical_fraction"
            if summary.elite_count == 0
            else "nominal_subset_product"
        )
    )
    parts.append(
        f"target_probability_at_level={summary.target_exceedance_probability:.6e}"
    )
    parts.append(f"target_reached={str(summary.target_reached).lower()}")
    if summary.n_fail > 0:
        parts.append(f"n_fail={summary.n_fail}")
    if summary.invalid_evaluations:
        parts.append(f"invalid={summary.invalid_evaluations}")
    print(" | ".join(parts), flush=True)


def _update_proposal_half_width(
    current_half_width: float,
    indicator_rate: float,
    target_indicator_acceptance: float,
    adaptation_gain: float,
    max_width_change_factor: float,
    min_proposal_half_width: float,
    max_proposal_half_width: float,
) -> float:
    error = indicator_rate - target_indicator_acceptance
    if abs(error) < 0.02:
        return float(
            min(max(current_half_width, min_proposal_half_width), max_proposal_half_width)
        )

    bounded_error = math.tanh(error / 0.15)
    step_multiplier = math.exp(0.5 * adaptation_gain * bounded_error)
    next_half_width = current_half_width * step_multiplier
    next_half_width = min(
        max(next_half_width, current_half_width / max_width_change_factor),
        current_half_width * max_width_change_factor,
    )
    return float(
        min(max(next_half_width, min_proposal_half_width), max_proposal_half_width)
    )


def _componentwise_metropolis_step(
    current: np.ndarray,
    rng: np.random.Generator,
    proposal_half_width: float,
) -> tuple[np.ndarray, bool]:
    proposal = np.asarray(current, dtype=float).copy()
    accepted_any = False

    for dim in range(proposal.size):
        current_coord = float(proposal[dim])
        proposed_coord = current_coord + float(rng.uniform(-proposal_half_width, proposal_half_width))
        delta = proposed_coord * proposed_coord - current_coord * current_coord
        log_alpha = -0.5 * delta
        if log_alpha >= 0.0 or math.log(max(float(rng.random()), _LOG_FLOOR)) < log_alpha:
            proposal[dim] = proposed_coord
            accepted_any = True

    return proposal, accepted_any


def _evaluate_points(points: np.ndarray, target: TargetEvaluator) -> tuple[np.ndarray, np.ndarray, int]:
    evaluate_batch = getattr(target, "evaluate_batch", None)
    if callable(evaluate_batch):
        values, log_targets, invalid = evaluate_batch(np.asarray(points, dtype=float))
        return (
            np.asarray(values, dtype=float),
            np.asarray(log_targets, dtype=float),
            int(invalid),
        )

    n_points = int(points.shape[0])
    values = np.empty(n_points, dtype=float)
    log_targets = np.empty(n_points, dtype=float)
    invalid = 0

    for idx in range(n_points):
        evaluation = _normalize_evaluation(target(np.asarray(points[idx], dtype=float)))
        values[idx] = evaluation.limit_state
        log_targets[idx] = evaluation.log_target
        if not evaluation.valid:
            invalid += 1

    return values, log_targets, invalid


def _initial_state(
    dimension: int,
    target: TargetEvaluator,
    samples_per_level: int,
    conditional_prob: float,
    max_levels: int,
    initial_scale: float,
    proposal_half_width: float,
    failure_threshold: float,
    level_zero_screen_samples: Optional[int],
    level_zero_screen_probability: float,
    rng: np.random.Generator,
    progress: bool,
    progress_callback: Optional[Callable[[str, LevelSummary], None]],
    evaluation_context_callback: Optional[Callable[[int], None]],
    trace_observer: Optional[SamplerTraceObserver],
) -> tuple[_EngineState, int, Optional[LevelZeroMonteCarloScreen]]:
    points = initial_scale * rng.standard_normal((samples_per_level, dimension))
    if evaluation_context_callback is not None:
        evaluation_context_callback(0)

    screen: Optional[LevelZeroMonteCarloScreen] = None
    screen_is_eligible = (
        max_levels > 0
        and level_zero_screen_samples is not None
        and samples_per_level >= int(level_zero_screen_samples)
    )
    if screen_is_eligible:
        checkpoint_samples = int(level_zero_screen_samples)
        checkpoint_values, checkpoint_log_targets, checkpoint_invalid = _evaluate_points(
            points[:checkpoint_samples],
            target,
        )
        observed_failures = int(
            np.count_nonzero(checkpoint_values <= float(failure_threshold))
        )
        required_failures = int(
            math.ceil(float(level_zero_screen_probability) * checkpoint_samples)
        )
        estimated_probability = observed_failures / float(checkpoint_samples)
        interval_lower, interval_upper = _wilson_score_interval(
            observed_failures,
            checkpoint_samples,
        )
        screen = LevelZeroMonteCarloScreen(
            checkpoint_samples=checkpoint_samples,
            probability_boundary=float(level_zero_screen_probability),
            required_failures=required_failures,
            observed_failures=observed_failures,
            estimated_probability=estimated_probability,
            confidence_level=0.95,
            confidence_interval_lower=interval_lower,
            confidence_interval_upper=interval_upper,
            triggered=(
                checkpoint_invalid == 0 and observed_failures >= required_failures
            ),
        )
        if screen.triggered:
            checkpoint_points = points[:checkpoint_samples]
            order = np.argsort(checkpoint_values, kind="stable")
            checkpoint_points = checkpoint_points[order]
            checkpoint_values = checkpoint_values[order]
            checkpoint_log_targets = checkpoint_log_targets[order]
            summary = LevelSummary(
                level=0,
                threshold=float(failure_threshold),
                min_g=float(checkpoint_values[0]),
                mean_g=float(np.mean(checkpoint_values)),
                max_g=float(checkpoint_values[-1]),
                n_fail=observed_failures,
                n_le_threshold=observed_failures,
                population_size=checkpoint_samples,
                population_unique_points=_count_unique_population_points(checkpoint_points),
                evaluations=checkpoint_samples,
                elite_count=0,
                chain_length=0,
                proposal_half_width=float(proposal_half_width),
                invalid_evaluations=0,
                proposed_moves=0,
                accepted_moves=0,
                indicator_acceptance_rate=None,
                probability_mass_before_level=1.0,
                threshold_exceedance_probability=estimated_probability,
                target_exceedance_probability=estimated_probability,
                target_reached=True,
            )
            state = _EngineState(
                points=checkpoint_points,
                values=checkpoint_values,
                log_targets=checkpoint_log_targets,
                total_evaluations=checkpoint_samples,
                level_summaries=[summary],
            )
            if trace_observer is not None:
                trace_observer.observe_population(
                    level=0,
                    points=checkpoint_points.copy(),
                    values=checkpoint_values.copy(),
                    log_targets=checkpoint_log_targets.copy(),
                    elite_count=0,
                    threshold=float(failure_threshold),
                )
            if progress:
                if progress_callback is not None:
                    progress_callback("subset", summary)
                else:
                    _print_level_progress("subset", summary)
            return state, 0, screen

        remaining_points = points[checkpoint_samples:]
        if int(remaining_points.shape[0]) > 0:
            remaining_values, remaining_log_targets, remaining_invalid = _evaluate_points(
                remaining_points,
                target,
            )
            values = np.concatenate((checkpoint_values, remaining_values))
            log_targets = np.concatenate((checkpoint_log_targets, remaining_log_targets))
            invalid = int(checkpoint_invalid + remaining_invalid)
        else:
            values = checkpoint_values
            log_targets = checkpoint_log_targets
            invalid = int(checkpoint_invalid)
    else:
        values, log_targets, invalid = _evaluate_points(points, target)

    order = np.argsort(values, kind="stable")
    points = points[order]
    values = values[order]
    log_targets = log_targets[order]

    elite_count = _elite_count(samples_per_level, conditional_prob)
    chain_length = _chain_length(samples_per_level, elite_count)
    threshold_index = elite_count - 1
    threshold = float(values[threshold_index])
    n_fail = int(np.sum(values <= float(failure_threshold)))
    threshold_probability = elite_count / float(samples_per_level)
    summary = LevelSummary(
        level=0,
        threshold=threshold,
        min_g=float(values[0]),
        mean_g=float(np.mean(values)),
        max_g=float(values[-1]),
        n_fail=n_fail,
        n_le_threshold=int(np.sum(values <= threshold)),
        population_size=int(points.shape[0]),
        population_unique_points=_count_unique_population_points(points),
        evaluations=samples_per_level,
        elite_count=elite_count,
        chain_length=chain_length,
        proposal_half_width=float(proposal_half_width),
        invalid_evaluations=invalid,
        proposed_moves=0,
        accepted_moves=0,
        indicator_acceptance_rate=None,
        probability_mass_before_level=1.0,
        threshold_exceedance_probability=threshold_probability,
        target_exceedance_probability=n_fail / float(samples_per_level),
        target_reached=threshold <= failure_threshold,
    )
    state = _EngineState(
        points=points,
        values=values,
        log_targets=log_targets,
        total_evaluations=samples_per_level,
        level_summaries=[summary],
    )
    if trace_observer is not None:
        trace_observer.observe_population(
            level=0,
            points=points.copy(),
            values=values.copy(),
            log_targets=log_targets.copy(),
            elite_count=elite_count,
            threshold=threshold,
        )
    if progress:
        if progress_callback is not None:
            progress_callback("subset", summary)
        else:
            _print_level_progress("subset", summary)
    return state, invalid, screen


def _run_levels(
    dimension: int,
    target: TargetEvaluator,
    state: _EngineState,
    samples_per_level: int,
    conditional_prob: float,
    max_levels: int,
    proposal_half_width: float,
    adapt_proposal_half_width: bool,
    target_indicator_acceptance: float,
    adaptation_gain: float,
    max_width_change_factor: float,
    min_proposal_half_width: float,
    max_proposal_half_width: float,
    failure_threshold: float,
    min_indicator_acceptance: float,
    rng: np.random.Generator,
    progress: bool,
    progress_callback: Optional[Callable[[str, LevelSummary], None]],
    evaluation_context_callback: Optional[Callable[[int], None]],
    trace_observer: Optional[SamplerTraceObserver],
) -> SamplerStopReason:
    elite_count = _elite_count(samples_per_level, conditional_prob)
    chain_length = _chain_length(samples_per_level, elite_count)
    current_proposal_half_width = float(proposal_half_width)
    stop_reason = SamplerStopReason.MAX_LEVELS

    for level in range(1, max_levels + 1):
        threshold_index = elite_count - 1
        threshold = float(state.values[threshold_index])
        if threshold <= failure_threshold:
            stop_reason = SamplerStopReason.CONVERGED
            break

        state.level_probabilities.append(elite_count / float(samples_per_level))

        seeds = state.points[:elite_count].copy()
        seeds_values = state.values[:elite_count].copy()
        seeds_log_targets = state.log_targets[:elite_count].copy()

        next_points = np.empty((samples_per_level, dimension), dtype=float)
        next_values = np.empty((samples_per_level,), dtype=float)
        next_log_targets = np.empty((samples_per_level,), dtype=float)
        next_points[:elite_count] = seeds
        next_values[:elite_count] = seeds_values
        next_log_targets[:elite_count] = seeds_log_targets
        total_proposed = 0
        indicator_accepted = 0
        invalid_level = 0

        current_points = seeds.copy()
        current_values = seeds_values.copy()
        current_log_targets = seeds_log_targets.copy()
        cursor = elite_count

        for wave_index in range(chain_length):
            proposals = np.empty_like(current_points)
            changed_mask = np.zeros((elite_count,), dtype=bool)

            for seed_index in range(elite_count):
                proposed_point, changed = _componentwise_metropolis_step(
                    current_points[seed_index],
                    rng,
                    current_proposal_half_width,
                )
                proposals[seed_index] = proposed_point
                changed_mask[seed_index] = changed

            total_proposed += elite_count

            proposed_values = current_values.copy()
            proposed_log_targets = current_log_targets.copy()
            if np.any(changed_mask):
                changed_indices = np.flatnonzero(changed_mask)
                if evaluation_context_callback is not None:
                    evaluation_context_callback(level)
                changed_values, changed_log_targets, invalid_changed = _evaluate_points(
                    proposals[changed_mask],
                    target,
                )
                proposed_values[changed_mask] = changed_values
                proposed_log_targets[changed_mask] = changed_log_targets
                state.total_evaluations += int(changed_indices.size)
                invalid_level += int(invalid_changed)

            event_mask = changed_mask & (proposed_values <= threshold)
            if trace_observer is not None:
                trace_observer.observe_transition_batch(
                    level=level,
                    wave_index=wave_index,
                    threshold=threshold,
                    proposal_half_width=current_proposal_half_width,
                    parent_points=current_points.copy(),
                    parent_values=current_values.copy(),
                    parent_log_targets=current_log_targets.copy(),
                    proposal_points=proposals.copy(),
                    proposal_values=proposed_values.copy(),
                    proposal_log_targets=proposed_log_targets.copy(),
                    changed_mask=changed_mask.copy(),
                    accepted_mask=event_mask.copy(),
                )
            accepted_indices = np.flatnonzero(event_mask)
            for seed_index in accepted_indices:
                current_points[seed_index] = proposals[seed_index]
                current_values[seed_index] = float(proposed_values[seed_index])
                current_log_targets[seed_index] = float(proposed_log_targets[seed_index])
            indicator_accepted += int(accepted_indices.size)

            if cursor < samples_per_level:
                take = min(samples_per_level - cursor, elite_count)
                next_points[cursor:cursor + take] = current_points[:take]
                next_values[cursor:cursor + take] = current_values[:take]
                next_log_targets[cursor:cursor + take] = current_log_targets[:take]
                cursor += take

        if cursor < samples_per_level:
            fill = samples_per_level - cursor
            next_points[cursor:] = seeds[:fill]
            next_values[cursor:] = seeds_values[:fill]
            next_log_targets[cursor:] = seeds_log_targets[:fill]

        order = np.argsort(next_values, kind="stable")
        state.points = next_points[order]
        state.values = next_values[order]
        state.log_targets = next_log_targets[order]
        state.completed_levels = level

        next_threshold = float(state.values[threshold_index])
        acceptance_rate = indicator_accepted / max(total_proposed, 1)
        probability_mass_before_level = float(
            np.prod(state.level_probabilities, dtype=float)
        )
        n_fail = int(np.sum(state.values <= float(failure_threshold)))
        summary = LevelSummary(
            level=level,
            threshold=next_threshold,
            min_g=float(state.values[0]),
            mean_g=float(np.mean(state.values)),
            max_g=float(state.values[-1]),
            n_fail=n_fail,
            n_le_threshold=int(np.sum(state.values <= next_threshold)),
            population_size=int(state.points.shape[0]),
            population_unique_points=_count_unique_population_points(state.points),
            evaluations=state.total_evaluations,
            elite_count=elite_count,
            chain_length=chain_length,
            proposal_half_width=current_proposal_half_width,
            invalid_evaluations=invalid_level,
            proposed_moves=total_proposed,
            accepted_moves=indicator_accepted,
            indicator_acceptance_rate=acceptance_rate,
            probability_mass_before_level=probability_mass_before_level,
            threshold_exceedance_probability=(
                probability_mass_before_level
                * elite_count
                / float(samples_per_level)
            ),
            target_exceedance_probability=(
                probability_mass_before_level
                * n_fail
                / float(samples_per_level)
            ),
            target_reached=next_threshold <= failure_threshold,
        )
        state.level_summaries.append(summary)
        if trace_observer is not None:
            trace_observer.observe_population(
                level=level,
                points=state.points.copy(),
                values=state.values.copy(),
                log_targets=state.log_targets.copy(),
                elite_count=elite_count,
                threshold=next_threshold,
            )
        if progress:
            if progress_callback is not None:
                progress_callback("subset", summary)
            else:
                _print_level_progress("subset", summary)

        if invalid_level > 0:
            stop_reason = SamplerStopReason.INVALID_EVALUATIONS
            break
        if acceptance_rate < min_indicator_acceptance:
            stop_reason = SamplerStopReason.LOW_ACCEPTANCE
            break
        if adapt_proposal_half_width:
            current_proposal_half_width = _update_proposal_half_width(
                current_half_width=current_proposal_half_width,
                indicator_rate=acceptance_rate,
                target_indicator_acceptance=target_indicator_acceptance,
                adaptation_gain=adaptation_gain,
                max_width_change_factor=max_width_change_factor,
                min_proposal_half_width=min_proposal_half_width,
                max_proposal_half_width=max_proposal_half_width,
            )

    else:
        threshold = float(state.values[elite_count - 1])
        if threshold <= failure_threshold:
            stop_reason = SamplerStopReason.CONVERGED

    return stop_reason


def run_subset_simulation_engine(
    dimension: int,
    target: TargetEvaluator,
    config: Optional[SubsetSimulationConfig] = None,
) -> SubsetSimulationResult:
    cfg = config or SubsetSimulationConfig()
    _validate_common_config(
        samples_per_level=int(cfg.samples_per_level),
        conditional_prob=float(cfg.conditional_prob),
        max_levels=int(cfg.max_levels),
        proposal_half_width=float(cfg.proposal_half_width),
        target_indicator_acceptance=float(cfg.target_indicator_acceptance),
        adaptation_gain=float(cfg.adaptation_gain),
        max_width_change_factor=float(cfg.max_width_change_factor),
        min_proposal_half_width=float(cfg.min_proposal_half_width),
        max_proposal_half_width=float(cfg.max_proposal_half_width),
        initial_scale=float(cfg.initial_scale),
        min_indicator_acceptance=float(cfg.min_indicator_acceptance),
        level_zero_screen_samples=(
            int(cfg.level_zero_screen_samples)
            if cfg.level_zero_screen_samples is not None
            else None
        ),
        level_zero_screen_probability=float(cfg.level_zero_screen_probability),
    )
    if cfg.progress:
        _print_run_config(
            method_name="subset",
            dimension=int(dimension),
            samples_per_level=int(cfg.samples_per_level),
            conditional_prob=float(cfg.conditional_prob),
            max_levels=int(cfg.max_levels),
            proposal_half_width=float(cfg.proposal_half_width),
            adapt_proposal_half_width=bool(cfg.adapt_proposal_half_width),
            target_indicator_acceptance=float(cfg.target_indicator_acceptance),
        )
    rng = np.random.default_rng(cfg.random_seed)
    state, invalid, level_zero_screen = _initial_state(
        dimension=int(dimension),
        target=target,
        samples_per_level=int(cfg.samples_per_level),
        conditional_prob=float(cfg.conditional_prob),
        max_levels=int(cfg.max_levels),
        initial_scale=float(cfg.initial_scale),
        proposal_half_width=float(cfg.proposal_half_width),
        failure_threshold=float(cfg.failure_threshold),
        level_zero_screen_samples=(
            int(cfg.level_zero_screen_samples)
            if cfg.level_zero_screen_samples is not None
            else None
        ),
        level_zero_screen_probability=float(cfg.level_zero_screen_probability),
        rng=rng,
        progress=bool(cfg.progress),
        progress_callback=cfg.progress_callback,
        evaluation_context_callback=cfg.evaluation_context_callback,
        trace_observer=cfg.trace_observer,
    )
    if level_zero_screen is not None and cfg.progress:
        decision = (
            "stop_direct_monte_carlo"
            if level_zero_screen.triggered
            else "continue_full_level_zero"
        )
        print(
            "[subset-screen] "
            f"checkpoint_n={level_zero_screen.checkpoint_samples} | "
            f"failures={level_zero_screen.observed_failures} | "
            f"required_failures={level_zero_screen.required_failures} | "
            f"p_hat={level_zero_screen.estimated_probability:.6f} | "
            f"ci95=[{level_zero_screen.confidence_interval_lower:.6f},"
            f"{level_zero_screen.confidence_interval_upper:.6f}] | "
            f"boundary={level_zero_screen.probability_boundary:.6f} | "
            f"decision={decision}",
            flush=True,
        )

    if invalid > 0:
        stop_reason = SamplerStopReason.INVALID_EVALUATIONS
    elif level_zero_screen is not None and level_zero_screen.triggered:
        stop_reason = SamplerStopReason.DIRECT_MONTE_CARLO_SCREEN
    else:
        stop_reason = _run_levels(
            dimension=int(dimension),
            target=target,
            state=state,
            samples_per_level=int(cfg.samples_per_level),
            conditional_prob=float(cfg.conditional_prob),
            max_levels=int(cfg.max_levels),
            proposal_half_width=float(cfg.proposal_half_width),
            adapt_proposal_half_width=bool(cfg.adapt_proposal_half_width),
            target_indicator_acceptance=float(cfg.target_indicator_acceptance),
            adaptation_gain=float(cfg.adaptation_gain),
            max_width_change_factor=float(cfg.max_width_change_factor),
            min_proposal_half_width=float(cfg.min_proposal_half_width),
            max_proposal_half_width=float(cfg.max_proposal_half_width),
            failure_threshold=float(cfg.failure_threshold),
            min_indicator_acceptance=float(cfg.min_indicator_acceptance),
            rng=rng,
            progress=bool(cfg.progress),
            progress_callback=cfg.progress_callback,
            evaluation_context_callback=cfg.evaluation_context_callback,
            trace_observer=cfg.trace_observer,
        )

    valid_count = int(state.values.size)
    failure_fraction = 0.0
    if valid_count > 0:
        failure_fraction = float(np.sum(state.values <= float(cfg.failure_threshold))) / float(valid_count)
    estimated_probability = float(np.prod(state.level_probabilities, dtype=float) * failure_fraction)
    terminal_summary = state.level_summaries[-1]
    reliable = stop_reason in {
        SamplerStopReason.CONVERGED,
        SamplerStopReason.DIRECT_MONTE_CARLO_SCREEN,
    }
    if stop_reason == SamplerStopReason.DIRECT_MONTE_CARLO_SCREEN:
        probability_method = "direct_monte_carlo_fixed_checkpoint"
    elif state.completed_levels == 0:
        probability_method = "direct_monte_carlo_full_level_zero"
    else:
        probability_method = "subset_simulation"

    return SubsetSimulationResult(
        stop_reason=stop_reason,
        total_evaluations=state.total_evaluations,
        completed_levels=state.completed_levels,
        level_summaries=state.level_summaries,
        final_population=state.points.copy(),
        final_limit_state=state.values.copy(),
        final_log_target=state.log_targets.copy(),
        level_probabilities=np.asarray(state.level_probabilities, dtype=float),
        estimated_probability=estimated_probability,
        reliable_probability=reliable,
        level_zero_screen=level_zero_screen,
        probability_method=probability_method,
        target_reached=bool(terminal_summary.target_reached),
        terminal_threshold=float(terminal_summary.threshold),
        terminal_threshold_probability=float(
            terminal_summary.threshold_exceedance_probability
        ),
    )
