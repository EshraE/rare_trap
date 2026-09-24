from __future__ import annotations

import numpy as np

from estimators_samplers.engine import (
    SamplerStopReason,
    SubsetSimulationConfig,
    _wilson_score_interval,
    run_subset_simulation_engine,
)
from estimators_samplers.targets import build_limit_state_target


class _CountingBatchTarget:
    def __init__(self, first_batch_failures: int = 0) -> None:
        self.first_batch_failures = int(first_batch_failures)
        self.batch_sizes: list[int] = []

    def evaluate_batch(self, points: np.ndarray):
        count = int(np.asarray(points).shape[0])
        values = np.ones(count, dtype=float)
        if not self.batch_sizes:
            values[: self.first_batch_failures] = -1.0
        self.batch_sizes.append(count)
        return values, np.zeros(count, dtype=float), 0


def _screen_config(*, max_levels: int = 1) -> SubsetSimulationConfig:
    return SubsetSimulationConfig(
        samples_per_level=400,
        conditional_prob=0.2,
        max_levels=max_levels,
        random_seed=17,
        progress=False,
        min_indicator_acceptance=0.0,
        level_zero_screen_samples=200,
        level_zero_screen_probability=0.1,
    )


def test_screen_stops_at_exact_twenty_of_two_hundred() -> None:
    target = _CountingBatchTarget(first_batch_failures=20)

    result = run_subset_simulation_engine(2, target, _screen_config())

    assert target.batch_sizes == [200]
    assert result.stop_reason == SamplerStopReason.DIRECT_MONTE_CARLO_SCREEN
    assert result.probability_method == "direct_monte_carlo_fixed_checkpoint"
    assert result.reliable_probability
    assert result.total_evaluations == 200
    assert result.completed_levels == 0
    assert result.estimated_probability == 0.1
    assert result.final_population.shape == (200, 2)
    assert result.level_summaries[0].elite_count == 0
    assert result.level_summaries[0].chain_length == 0
    assert result.level_zero_screen is not None
    assert result.level_zero_screen.triggered
    assert result.level_zero_screen.required_failures == 20
    assert result.level_zero_screen.observed_failures == 20
    assert result.target_reached
    assert result.terminal_threshold == 0.0
    assert result.terminal_threshold_probability == 0.1
    assert result.level_summaries[0].threshold_exceedance_probability == 0.1
    assert result.level_summaries[0].target_exceedance_probability == 0.1


def test_screen_below_boundary_completes_level_zero_before_subset() -> None:
    target = _CountingBatchTarget(first_batch_failures=19)

    result = run_subset_simulation_engine(2, target, _screen_config())

    assert target.batch_sizes[:2] == [200, 200]
    assert result.level_summaries[0].population_size == 400
    assert result.level_summaries[0].evaluations == 400
    assert result.level_zero_screen is not None
    assert not result.level_zero_screen.triggered
    assert result.level_zero_screen.observed_failures == 19


def test_direct_monte_carlo_run_ignores_subset_screen() -> None:
    target = _CountingBatchTarget(first_batch_failures=300)
    config = _screen_config(max_levels=0)
    config.samples_per_level = 300

    result = run_subset_simulation_engine(2, target, config)

    assert target.batch_sizes == [300]
    assert result.total_evaluations == 300
    assert result.level_zero_screen is None
    assert result.probability_method == "direct_monte_carlo_full_level_zero"


def test_non_triggering_screen_preserves_subset_trajectory() -> None:
    def first_coordinate(points: np.ndarray):
        points = np.asarray(points)
        return float(points[0]) if points.ndim == 1 else points[:, 0]

    common = dict(
        samples_per_level=400,
        conditional_prob=0.2,
        max_levels=1,
        failure_threshold=-10.0,
        random_seed=29,
        progress=False,
        min_indicator_acceptance=0.0,
    )
    target = build_limit_state_target(2, first_coordinate)
    unscreened = run_subset_simulation_engine(
        2,
        target,
        SubsetSimulationConfig(**common),
    )
    screened = run_subset_simulation_engine(
        2,
        target,
        SubsetSimulationConfig(
            **common,
            level_zero_screen_samples=200,
            level_zero_screen_probability=0.1,
        ),
    )

    assert screened.level_zero_screen is not None
    assert not screened.level_zero_screen.triggered
    np.testing.assert_array_equal(screened.final_population, unscreened.final_population)
    np.testing.assert_array_equal(screened.final_limit_state, unscreened.final_limit_state)
    assert screened.estimated_probability == unscreened.estimated_probability


def test_level_threshold_probabilities_follow_subset_products() -> None:
    def positive_limit_state(points: np.ndarray):
        points = np.asarray(points)
        count = 1 if points.ndim == 1 else int(points.shape[0])
        values = np.ones(count, dtype=float)
        return float(values[0]) if points.ndim == 1 else values

    result = run_subset_simulation_engine(
        2,
        build_limit_state_target(2, positive_limit_state),
        SubsetSimulationConfig(
            samples_per_level=20,
            conditional_prob=0.2,
            max_levels=1,
            failure_threshold=-1.0,
            random_seed=31,
            progress=False,
            min_indicator_acceptance=0.0,
        ),
    )

    initial, conditional = result.level_summaries
    assert initial.probability_mass_before_level == 1.0
    assert initial.threshold_exceedance_probability == 0.2
    assert initial.target_exceedance_probability == 0.0
    assert conditional.probability_mass_before_level == 0.2
    np.testing.assert_allclose(conditional.threshold_exceedance_probability, 0.04)
    assert conditional.target_exceedance_probability == 0.0
    assert not result.target_reached
    np.testing.assert_allclose(result.terminal_threshold_probability, 0.04)
    assert result.estimated_probability == conditional.target_exceedance_probability


def test_threshold_trace_uses_configured_conditional_probability() -> None:
    def positive_limit_state(points: np.ndarray):
        points = np.asarray(points)
        return 1.0 if points.ndim == 1 else np.ones(points.shape[0], dtype=float)

    result = run_subset_simulation_engine(
        2,
        build_limit_state_target(2, positive_limit_state),
        SubsetSimulationConfig(
            samples_per_level=20,
            conditional_prob=0.3,
            max_levels=1,
            failure_threshold=-1.0,
            random_seed=37,
            progress=False,
            min_indicator_acceptance=0.0,
        ),
    )

    initial, conditional = result.level_summaries
    assert initial.elite_count == 6
    np.testing.assert_allclose(result.level_probabilities, [0.3])
    np.testing.assert_allclose(initial.threshold_exceedance_probability, 0.3)
    np.testing.assert_allclose(conditional.probability_mass_before_level, 0.3)
    np.testing.assert_allclose(conditional.threshold_exceedance_probability, 0.09)


def test_wilson_interval_for_twenty_of_two_hundred() -> None:
    lower, upper = _wilson_score_interval(20, 200)

    np.testing.assert_allclose(lower, 0.06567044866909588, rtol=0, atol=1e-15)
    np.testing.assert_allclose(upper, 0.1494058124327174, rtol=0, atol=1e-15)
