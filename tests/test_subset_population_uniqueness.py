from __future__ import annotations

import unittest

import numpy as np

from estimators_samplers.engine import (
    SubsetSimulationConfig,
    run_subset_simulation_engine,
)
from estimators_samplers.targets import build_limit_state_target


class PopulationUniquenessTest(unittest.TestCase):
    def test_rejected_moves_reduce_retained_population_uniqueness(self) -> None:
        def limit_state(points: np.ndarray):
            values = np.asarray(points)
            if values.ndim == 1:
                return float(values[0])
            return values[:, 0]

        result = run_subset_simulation_engine(
            dimension=1,
            target=build_limit_state_target(1, limit_state),
            config=SubsetSimulationConfig(
                samples_per_level=10,
                conditional_prob=0.2,
                max_levels=1,
                failure_threshold=-10.0,
                random_seed=7,
                progress=False,
            ),
        )

        initial, conditional = result.level_summaries
        exact_final_unique = int(np.unique(result.final_population, axis=0).shape[0])

        self.assertEqual(initial.population_unique_points, 10)
        self.assertEqual(conditional.population_unique_points, exact_final_unique)
        self.assertLess(conditional.population_unique_points, conditional.population_size)
        self.assertGreater(conditional.evaluations, conditional.population_unique_points)

    def test_evaluation_context_callback_reports_subset_levels(self) -> None:
        observed_levels: list[int] = []

        def limit_state(points: np.ndarray):
            values = np.asarray(points)
            if values.ndim == 1:
                return float(values[0])
            return values[:, 0]

        run_subset_simulation_engine(
            dimension=1,
            target=build_limit_state_target(1, limit_state),
            config=SubsetSimulationConfig(
                samples_per_level=10,
                conditional_prob=0.2,
                max_levels=2,
                failure_threshold=-10.0,
                random_seed=7,
                progress=False,
                evaluation_context_callback=observed_levels.append,
            ),
        )

        self.assertEqual(observed_levels[0], 0)
        self.assertIn(1, observed_levels)


if __name__ == "__main__":
    unittest.main()
