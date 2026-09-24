from __future__ import annotations

import numpy as np

from estimators_samplers.engine import SubsetSimulationConfig, run_subset_simulation_engine
from estimators_samplers.targets import build_limit_state_target


def test_discrete_ties_use_stable_sorting_without_auxiliary_state() -> None:
    def constant_limit_state(points: np.ndarray):
        points = np.asarray(points)
        if points.ndim == 1:
            return 1.0
        return np.ones(points.shape[0], dtype=float)

    result = run_subset_simulation_engine(
        dimension=2,
        target=build_limit_state_target(2, constant_limit_state),
        config=SubsetSimulationConfig(
            samples_per_level=10,
            conditional_prob=0.2,
            max_levels=1,
            random_seed=12,
            progress=False,
            min_indicator_acceptance=0.0,
        ),
    )

    assert result.level_summaries[0].threshold == 1.0
    assert result.level_summaries[0].n_le_threshold == 10
    assert result.level_summaries[1].elite_count == 2
    assert result.level_probabilities.tolist() == [0.2]


def test_intermediate_boundary_is_mth_sorted_value() -> None:
    def first_coordinate(points: np.ndarray):
        points = np.asarray(points)
        return float(points[0]) if points.ndim == 1 else points[:, 0]

    result = run_subset_simulation_engine(
        dimension=1,
        target=build_limit_state_target(1, first_coordinate),
        config=SubsetSimulationConfig(
            samples_per_level=20,
            conditional_prob=0.2,
            max_levels=0,
            random_seed=4,
            progress=False,
        ),
    )

    assert result.level_summaries[0].elite_count == 4
    assert result.level_summaries[0].threshold == result.final_limit_state[3]


def test_configured_failure_threshold_drives_summary_and_probability() -> None:
    def first_coordinate(points: np.ndarray):
        points = np.asarray(points)
        return float(points[0]) if points.ndim == 1 else points[:, 0]

    result = run_subset_simulation_engine(
        dimension=1,
        target=build_limit_state_target(1, first_coordinate),
        config=SubsetSimulationConfig(
            samples_per_level=20,
            conditional_prob=0.2,
            max_levels=0,
            failure_threshold=0.5,
            random_seed=9,
            progress=False,
        ),
    )

    expected = int(np.count_nonzero(result.final_limit_state <= 0.5))
    assert result.level_summaries[0].n_fail == expected
    assert result.estimated_probability == expected / 20.0


def test_same_seed_reproduces_population_and_values() -> None:
    def squared_radius(points: np.ndarray):
        points = np.asarray(points)
        return float(np.sum(points * points)) if points.ndim == 1 else np.sum(points * points, axis=1)

    config = SubsetSimulationConfig(
        samples_per_level=20,
        conditional_prob=0.2,
        max_levels=1,
        random_seed=77,
        progress=False,
        min_indicator_acceptance=0.0,
    )
    target = build_limit_state_target(2, squared_radius)
    first = run_subset_simulation_engine(2, target, config)
    second = run_subset_simulation_engine(2, target, config)

    np.testing.assert_array_equal(first.final_population, second.final_population)
    np.testing.assert_array_equal(first.final_limit_state, second.final_limit_state)
