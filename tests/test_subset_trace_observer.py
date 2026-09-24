from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from estimators_samplers import SubsetSimulationConfig, SubsetTraceObserver
from estimators_samplers.engine import run_subset_simulation_engine
from estimators_samplers.targets import build_limit_state_target
from utils.config import PROJECTION_SEED
from utils.subset_trace_reporting import save_subset_trace


def _run(observer: SubsetTraceObserver | None = None):
    def limit_state(points: np.ndarray):
        values = np.asarray(points)
        if values.ndim == 1:
            return float(values[0] + 0.1 * values[1])
        return values[:, 0] + 0.1 * values[:, 1]

    return run_subset_simulation_engine(
        dimension=2,
        target=build_limit_state_target(2, limit_state),
        config=SubsetSimulationConfig(
            samples_per_level=20,
            conditional_prob=0.2,
            max_levels=2,
            failure_threshold=-10.0,
            random_seed=17,
            progress=False,
            trace_observer=observer,
        ),
    )


class SubsetTraceObserverTest(unittest.TestCase):
    def test_observer_does_not_change_sampler_result(self) -> None:
        baseline = _run()
        observer = SubsetTraceObserver()
        observed = _run(observer)

        np.testing.assert_array_equal(observed.final_population, baseline.final_population)
        np.testing.assert_array_equal(observed.final_limit_state, baseline.final_limit_state)
        np.testing.assert_array_equal(observed.final_log_target, baseline.final_log_target)
        np.testing.assert_array_equal(observed.level_probabilities, baseline.level_probabilities)
        self.assertEqual(observed.level_summaries, baseline.level_summaries)
        self.assertEqual(observed.stop_reason, baseline.stop_reason)
        self.assertEqual(observed.total_evaluations, baseline.total_evaluations)
        self.assertEqual(observed.estimated_probability, baseline.estimated_probability)

    def test_observer_records_complete_lineage_and_population_counts(self) -> None:
        observer = SubsetTraceObserver()
        result = _run(observer)

        self.assertEqual(len(observer.population_summaries), len(result.level_summaries))
        self.assertEqual(
            len(observer.population_slots),
            sum(summary.population_size for summary in result.level_summaries),
        )
        self.assertEqual(
            len(observer.transitions),
            sum(summary.proposed_moves for summary in result.level_summaries),
        )
        self.assertEqual(
            sum(bool(row["accepted"]) for row in observer.transitions),
            sum(summary.accepted_moves for summary in result.level_summaries),
        )
        self.assertEqual(
            sum(bool(row["model_evaluated"]) for row in observer.transitions),
            result.total_evaluations - result.level_summaries[0].population_size,
        )

        state_ids = {str(row["state_id"]) for row in observer.states}
        first_seen_order = {
            str(row["state_id"]): index for index, row in enumerate(observer.states)
        }
        for row in observer.population_slots:
            self.assertIn(str(row["state_id"]), state_ids)
        for row in observer.transitions:
            self.assertIn(str(row["parent_state_id"]), state_ids)
            self.assertIn(str(row["result_state_id"]), state_ids)
            if row["proposal_state_id"] is not None:
                self.assertTrue(row["accepted"])
                self.assertLess(
                    first_seen_order[str(row["parent_state_id"])],
                    first_seen_order[str(row["proposal_state_id"])],
                )

        for summary, observed_population in zip(
            result.level_summaries, observer.population_summaries
        ):
            self.assertEqual(
                observed_population["distinct_state_count"],
                summary.population_unique_points,
            )

    def test_trace_writer_joins_evaluations_to_parent_states(self) -> None:
        observer = SubsetTraceObserver()
        result = _run(observer)
        initial_states = [
            row for row in observer.states if int(row["first_seen_subset_level"]) == 0
        ]
        transition_by_ordinal = {
            int(row["evaluation_ordinal"]): row
            for row in observer.transitions
            if row["evaluation_ordinal"] is not None
        }

        evaluations = []
        for ordinal in range(result.total_evaluations):
            if ordinal < len(initial_states):
                state = initial_states[ordinal]
                point_hash = state["latent_hash"]
                limit_state = float(state["limit_state"])
                level = 0
            else:
                transition = transition_by_ordinal[ordinal]
                point_hash = transition["proposal_latent_hash"]
                limit_state = float(transition["proposal_limit_state"])
                level = int(transition["subset_level"])
            score = 100.0 - limit_state
            length = max(1, int(round(score)))
            repetition = min(1.0, length / 200.0)
            evaluations.append(
                {
                    "sample_index": ordinal + 1,
                    "subset_level": level,
                    "output_tokens": length,
                    "output_token_margin": length - 100,
                    "performance_metric": "output_length",
                    "performance_score": score,
                    "performance_threshold": 100.0,
                    "performance_margin": score - 100.0,
                    "repetition_score": repetition,
                    "rep_2": repetition,
                    "rep_3": repetition / 2.0,
                    "rep_4": repetition / 3.0,
                    "success": score >= 100.0,
                    "hit_cap": False,
                    "elapsed_sec": 0.01,
                    "tokens_per_second": 1000.0,
                    "input_token_count": 2,
                    "input_token_ids": [ordinal, ordinal + 1],
                    "input_text": f"prompt-{ordinal}",
                    "input_hash": f"input-{ordinal}",
                    "output_hash": f"output-{ordinal}",
                    "latent_hash": point_hash,
                    "repeat_count": 1,
                    "repeat_aggregation": "single",
                    "repeat_lengths": [],
                }
            )

        obj = SimpleNamespace(
            evaluation_trace=evaluations,
            threshold=100,
            performance_threshold=100.0,
            performance_metric="output_length",
            repetition_score_threshold=0.9,
            run_metadata={"projection_seed": PROJECTION_SEED},
            generator=SimpleNamespace(max_new_tokens=103, device=None),
        )
        args = SimpleNamespace(
            tested_llm_model="test/model",
            surrogate_token_model="test/surrogate",
            tested_model_revision=None,
            surrogate_model_revision=None,
            seed=17,
            projection_mode="factorized-embedding-whitened",
            input_prompt_len=2,
            ss_sample_space_dim=2,
            ss_samples_per_level=20,
            ss_conditional_prob=0.2,
            ss_max_levels=2,
            llm_output_len_threshold=100,
            repetition_score_threshold=0.9,
            llm_output_sampler="greedy",
            chat_template_thinking="auto",
            generation_batch_size=4,
            objective_repeats=1,
            objective_aggregation="mean",
        )

        with tempfile.TemporaryDirectory() as directory:
            save_subset_trace(
                path=directory,
                observer=observer,
                obj=obj,
                args=args,
                result=result,
            )
            output_dir = Path(directory)
            manifest = json.loads((output_dir / "manifest.json").read_text())
            levels = json.loads((output_dir / "levels.json").read_text())["levels"]
            enriched = [
                json.loads(line)
                for line in (output_dir / "evaluations.jsonl").read_text().splitlines()
            ]
            transitions = [
                json.loads(line)
                for line in (output_dir / "transitions.jsonl").read_text().splitlines()
            ]
            states = [
                json.loads(line)
                for line in (output_dir / "states.jsonl").read_text().splitlines()
            ]
            population = [
                json.loads(line)
                for line in (output_dir / "population_slots.jsonl").read_text().splitlines()
            ]
            vectors = np.load(output_dir / "latent_states.npz")

        self.assertEqual(manifest["record_counts"]["evaluations"], result.total_evaluations)
        self.assertEqual(manifest["estimator_seed"], 17)
        self.assertEqual(manifest["projection_seed"], PROJECTION_SEED)
        self.assertEqual(len(levels), len(result.level_summaries))
        self.assertEqual(len(enriched), result.total_evaluations)
        self.assertEqual(len(transitions), len(observer.transitions))
        self.assertEqual(len(states), len(observer.states))
        self.assertEqual(len(population), len(observer.population_slots))
        self.assertEqual(vectors["latent_vectors"].shape[0], len(states))

        conditional = [row for row in enriched if row["evaluation_role"] == "conditional_proposal"]
        self.assertTrue(conditional)
        self.assertTrue(all(row["parent_state_id"] is not None for row in conditional))
        self.assertTrue(all(row["parent_output_tokens"] is not None for row in conditional))
        self.assertEqual(
            sum(bool(row["accepted"]) for row in transitions),
            sum(summary.accepted_moves for summary in result.level_summaries),
        )


if __name__ == "__main__":
    unittest.main()
