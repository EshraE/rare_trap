from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from estimators_samplers.engine import LevelSummary
from utils.config import PROJECTION_SEED
from utils.reporting import (
    _final_population_exceedances_payload,
    _level_zero_screen_payload,
    _threshold_probability_trace,
    print_final_recap,
    print_level_progress,
    save_all_evaluations,
    save_worst_cases,
)
from utils.thinktrap_objective import EvaluationRecord


class SuccessReportingTest(unittest.TestCase):
    def test_save_all_evaluations_records_seed_roles(self) -> None:
        obj = SimpleNamespace(
            evaluation_trace=[{"sample_index": 1, "subset_level": 0, "output_tokens": 12}],
            performance_metric="output_length",
            run_metadata={"projection_seed": PROJECTION_SEED},
        )
        args = argparse.Namespace(
            tested_llm_model="tested",
            surrogate_token_model="surrogate",
            tested_model_revision=None,
            surrogate_model_revision=None,
            seed=29,
            projection_mode="factorized-embedding-whitened",
            input_prompt_len=40,
            ss_sample_space_dim=200,
            ss_samples_per_level=500,
            ss_conditional_prob=0.1,
            ss_max_levels=4,
            llm_output_len_threshold=6000,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.json"
            with redirect_stdout(io.StringIO()):
                save_all_evaluations(path=str(path), obj=obj, args=args)
            payload = json.loads(path.read_text())

        self.assertEqual(payload["metadata"]["estimator_seed"], 29)
        self.assertEqual(payload["metadata"]["projection_seed"], PROJECTION_SEED)
        self.assertEqual(payload["metadata"]["record_count"], 1)
        self.assertEqual(payload["records"][0]["output_tokens"], 12)

    def test_level_progress_reports_output_lengths_and_successes(self) -> None:
        summary = LevelSummary(
            level=1,
            threshold=7663.0,
            min_g=4832.0,
            mean_g=7721.71,
            max_g=7887.0,
            n_fail=3,
            n_le_threshold=40,
            population_size=200,
            population_unique_points=90,
            evaluations=360,
            elite_count=40,
            chain_length=4,
            proposal_half_width=1.0,
            proposed_moves=160,
            accepted_moves=40,
            indicator_acceptance_rate=0.25,
        )
        obj = SimpleNamespace(
            threshold=8000,
            unique_input_hashes=set(range(360)),
            unique_z_hashes=set(range(360)),
            unique_success_input_hashes=set(range(3)),
        )

        output = io.StringIO()
        with redirect_stdout(output):
            print_level_progress("subset", summary, obj)
        line = output.getvalue()

        self.assertIn("level_output_token_threshold=337.0", line)
        self.assertIn("min_output_tokens=113.0", line)
        self.assertIn("mean_output_tokens=278.3", line)
        self.assertIn("max_output_tokens=3168.0", line)
        self.assertIn("n_success=3", line)
        self.assertIn("population_unique_z=90", line)
        self.assertNotIn("min_g=", line)
        self.assertNotIn("n_fail=", line)

    def test_saved_cases_use_success_and_output_length_schema(self) -> None:
        record = EvaluationRecord(
            sample_index=4,
            subset_level=2,
            length=503,
            output_token_margin=3,
            success=True,
            hit_cap=True,
            elapsed_sec=1.0,
            tokens_per_second=503.0,
            input_token_count=10,
            rendered_input_text="test input",
            completion_text="test output",
        )
        obj = SimpleNamespace(
            best_records=[record],
            unique_input_hashes={b"input"},
            unique_success_input_hashes={b"input"},
            unique_z_hashes={b"z"},
            record_count=1,
            generation_batch_size=1,
        )
        args = argparse.Namespace(
            tested_llm_model="tested",
            surrogate_token_model="surrogate",
            llm_output_sampler="sample",
            generation_batch_size=1,
            ss_samples_per_level=10,
            ss_max_levels=1,
            input_prompt_len=5,
            ss_sample_space_dim=10,
            projection_mode="factorized-embedding-whitened",
            covariance_eps=1e-5,
            llm_output_len_threshold=500,
            seed=0,
            case_output_text_count=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            with redirect_stdout(io.StringIO()):
                result = SimpleNamespace(
                    stop_reason=SimpleNamespace(value="converged"),
                    reliable_probability=True,
                    estimated_probability=0.125,
                    completed_levels=1,
                    total_evaluations=18,
                    level_probabilities=np.asarray([0.2]),
                    final_limit_state=np.asarray([-1.0, 2.0]),
                    level_summaries=[],
                )
                save_worst_cases(
                    path=str(path),
                    obj=obj,
                    args=args,
                    case_count=1,
                    result=result,
                )
            payload = json.loads(path.read_text())

        self.assertEqual(payload["metadata"]["unique_success_input_text_count"], 1)
        self.assertEqual(payload["metadata"]["generation_batch_size"], 1)
        self.assertEqual(payload["metadata"]["requested_generation_batch_size"], 1)
        self.assertEqual(payload["metadata"]["artifact_schema_version"], 5)
        self.assertEqual(
            payload["metadata"]["estimator_implementation"],
            "stable_order_statistic_fixed_level0_screen_v3",
        )
        self.assertEqual(payload["metadata"]["estimator_result"]["stop_reason"], "converged")
        self.assertEqual(payload["metadata"]["estimator_result"]["final_failure_count"], 1)
        case = payload["cases"][0]
        self.assertEqual(case["subset_level"], 2)
        self.assertEqual(case["output_tokens"], 503)
        self.assertEqual(case["output_token_margin"], 3)
        self.assertTrue(case["success"])
        self.assertTrue(case["output_text_saved"])
        self.assertEqual(case["output_text"], "test output")
        self.assertEqual(case["repeat_count"], 1)
        self.assertEqual(case["repeat_aggregation"], "single")
        self.assertEqual(case["repeat_lengths"], [])
        self.assertIsNone(case["best_repeat_index"])
        self.assertIn("level=2", case["case_line"])
        self.assertNotIn("g_value", case)
        self.assertNotIn("reached_threshold", case)

    def test_final_population_exceedances_report_both_metrics_for_every_hit(self) -> None:
        metrics_by_point = {
            (1.0, 2.0): {
                "source_sample_index": 7,
                "source_subset_level": 2,
                "output_tokens": 20003,
                "repetition_score": 0.998,
                "rep_2": 0.95,
                "rep_3": 0.94,
                "rep_4": 0.93,
                "selected_performance_metric": "output_length",
                "selected_performance_score": 20003.0,
                "selected_performance_threshold": 20000.0,
                "selected_threshold_exceeded": True,
                "hit_generation_cap": True,
            }
        }
        obj = SimpleNamespace(
            performance_metric="output_length",
            performance_threshold=20000.0,
            metrics_for_latent_point=lambda point: metrics_by_point.get(tuple(point)),
        )
        result = SimpleNamespace(
            final_population=np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]),
            final_limit_state=np.asarray([-3.0, -3.0, 1.0]),
        )

        payload = _final_population_exceedances_payload(result, obj)

        self.assertEqual(payload["exceedance_count"], 2)
        self.assertEqual(payload["matched_metric_count"], 2)
        self.assertEqual(payload["missing_metric_count"], 0)
        self.assertTrue(payload["duplicate_retained_rows_preserved"])
        self.assertEqual([row["final_population_index"] for row in payload["rows"]], [0, 1])
        self.assertEqual([row["output_tokens"] for row in payload["rows"]], [20003, 20003])
        self.assertEqual([row["repetition_score"] for row in payload["rows"]], [0.998, 0.998])

    def test_direct_monte_carlo_screen_reporting_is_explicit(self) -> None:
        screen = SimpleNamespace(
            checkpoint_samples=200,
            probability_boundary=0.1,
            required_failures=20,
            observed_failures=24,
            estimated_probability=0.12,
            confidence_level=0.95,
            confidence_interval_lower=0.082,
            confidence_interval_upper=0.173,
            triggered=True,
        )
        result = SimpleNamespace(
            stop_reason=SimpleNamespace(value="direct_monte_carlo_screen"),
            completed_levels=0,
            total_evaluations=200,
            estimated_probability=0.12,
            reliable_probability=True,
            final_limit_state=np.asarray([-1.0, 2.0]),
            level_zero_screen=screen,
            level_summaries=[
                LevelSummary(
                    level=0,
                    threshold=0.0,
                    min_g=-1.0,
                    mean_g=0.5,
                    max_g=2.0,
                    n_fail=24,
                    n_le_threshold=24,
                    population_size=200,
                    population_unique_points=200,
                    evaluations=200,
                    elite_count=0,
                    chain_length=0,
                    proposal_half_width=1.0,
                    threshold_exceedance_probability=0.12,
                    target_exceedance_probability=0.12,
                    target_reached=True,
                )
            ],
            target_reached=True,
            terminal_threshold_probability=0.12,
        )

        payload = _level_zero_screen_payload(result)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["checkpoint_samples"], 200)
        self.assertEqual(payload["observed_failures"], 24)
        self.assertEqual(payload["confidence_method"], "wilson_score")
        self.assertEqual(payload["decision"], "stop_and_skip_conditional_levels")

        output = io.StringIO()
        with redirect_stdout(output):
            print_final_recap("subset", result, threshold=100)
        line = output.getvalue()
        self.assertIn("probability_method=direct_monte_carlo_fixed_checkpoint", line)
        self.assertIn("successes=24/200", line)
        self.assertIn("ci95=[0.082000,0.173000]", line)
        self.assertIn("conditional_subset_levels_skipped=true", line)

    def test_threshold_probability_trace_is_plot_ready(self) -> None:
        summaries = [
            LevelSummary(
                level=0,
                threshold=500.0,
                min_g=100.0,
                mean_g=700.0,
                max_g=900.0,
                n_fail=0,
                n_le_threshold=42,
                population_size=200,
                population_unique_points=200,
                evaluations=200,
                elite_count=40,
                chain_length=4,
                proposal_half_width=1.0,
                probability_mass_before_level=1.0,
                threshold_exceedance_probability=0.2,
                target_exceedance_probability=0.0,
                target_reached=False,
            ),
            LevelSummary(
                level=1,
                threshold=200.0,
                min_g=50.0,
                mean_g=400.0,
                max_g=700.0,
                n_fail=0,
                n_le_threshold=40,
                population_size=200,
                population_unique_points=150,
                evaluations=360,
                elite_count=40,
                chain_length=4,
                proposal_half_width=1.0,
                probability_mass_before_level=0.2,
                threshold_exceedance_probability=0.04,
                target_exceedance_probability=0.0,
                target_reached=False,
            ),
        ]

        trace = _threshold_probability_trace(
            summaries,
            1000.0,
            performance_metric="output_length",
        )

        self.assertEqual([item["level"] for item in trace], [0, 1])
        self.assertEqual(
            [item["achieved_performance_threshold"] for item in trace],
            [500.0, 800.0],
        )
        self.assertEqual(
            [item["threshold_exceedance_probability"] for item in trace],
            [0.2, 0.04],
        )
        self.assertEqual(trace[0]["empirical_conditional_threshold_fraction"], 0.21)
        self.assertEqual(trace[0]["threshold_probability_basis"], "nominal_subset_product")

    def test_saved_cases_rank_successes_before_longer_non_hits(self) -> None:
        records = [
            EvaluationRecord(
                sample_index=1,
                subset_level=0,
                length=999,
                output_token_margin=-1,
                success=False,
                hit_cap=False,
                elapsed_sec=1.0,
                tokens_per_second=999.0,
                input_token_count=10,
                rendered_input_text="long non-hit",
                completion_text="long non-hit output",
            ),
            EvaluationRecord(
                sample_index=2,
                subset_level=0,
                length=500,
                output_token_margin=0,
                success=True,
                hit_cap=True,
                elapsed_sec=1.0,
                tokens_per_second=500.0,
                input_token_count=10,
                rendered_input_text="short hit",
                completion_text="short hit output",
                repeat_lengths=[400, 600],
                repeat_aggregation="mean",
                repeat_count=2,
                best_repeat_index=1,
            ),
            EvaluationRecord(
                sample_index=3,
                subset_level=0,
                length=300,
                output_token_margin=-200,
                success=False,
                hit_cap=False,
                elapsed_sec=1.0,
                tokens_per_second=300.0,
                input_token_count=10,
                rendered_input_text="short non-hit",
                completion_text="short non-hit output",
            ),
        ]
        obj = SimpleNamespace(
            best_records=records,
            unique_input_hashes={b"a", b"b", b"c"},
            unique_success_input_hashes={b"b"},
            unique_z_hashes={b"za", b"zb", b"zc"},
            record_count=3,
            generation_batch_size=1,
        )
        args = argparse.Namespace(
            tested_llm_model="tested",
            surrogate_token_model="surrogate",
            llm_output_sampler="greedy",
            generation_batch_size=1,
            ss_samples_per_level=10,
            ss_max_levels=1,
            input_prompt_len=5,
            ss_sample_space_dim=10,
            projection_mode="factorized-embedding-whitened",
            covariance_eps=1e-5,
            llm_output_len_threshold=500,
            seed=0,
            case_output_text_count=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            with redirect_stdout(io.StringIO()):
                save_worst_cases(path=str(path), obj=obj, args=args, case_count=2)
            payload = json.loads(path.read_text())

        self.assertEqual(payload["metadata"]["case_rank_policy"], "success_first_then_output_tokens")
        self.assertEqual(payload["metadata"]["saved_count"], 2)
        self.assertEqual(payload["metadata"]["saved_success_count"], 1)
        self.assertTrue(payload["cases"][0]["success"])
        self.assertEqual(payload["cases"][0]["input_text"], "short hit")
        self.assertEqual(payload["cases"][0]["repeat_count"], 2)
        self.assertEqual(payload["cases"][0]["repeat_aggregation"], "mean")
        self.assertEqual(payload["cases"][0]["repeat_lengths"], [400, 600])
        self.assertEqual(payload["cases"][0]["best_repeat_index"], 1)
        self.assertFalse(payload["cases"][1]["success"])
        self.assertEqual(payload["cases"][1]["input_text"], "long non-hit")

    def test_saved_cases_can_omit_most_output_text(self) -> None:
        records = [
            EvaluationRecord(
                sample_index=index + 1,
                subset_level=0,
                length=100 - index,
                output_token_margin=-index,
                success=False,
                hit_cap=False,
                elapsed_sec=1.0,
                tokens_per_second=100.0,
                input_token_count=10,
                rendered_input_text=f"input {index}",
                completion_text=(f"output {index}" if index == 0 else None),
            )
            for index in range(3)
        ]
        obj = SimpleNamespace(
            best_records=records,
            unique_input_hashes={b"a", b"b", b"c"},
            unique_success_input_hashes=set(),
            unique_z_hashes={b"za", b"zb", b"zc"},
            record_count=3,
            generation_batch_size=1,
        )
        args = argparse.Namespace(
            tested_llm_model="tested",
            surrogate_token_model="surrogate",
            llm_output_sampler="greedy",
            generation_batch_size=1,
            ss_samples_per_level=10,
            ss_max_levels=1,
            input_prompt_len=5,
            ss_sample_space_dim=10,
            projection_mode="factorized-embedding-whitened",
            covariance_eps=1e-5,
            llm_output_len_threshold=500,
            seed=0,
            case_output_text_count=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            with redirect_stdout(io.StringIO()):
                save_worst_cases(path=str(path), obj=obj, args=args, case_count=3)
            payload = json.loads(path.read_text())

        self.assertEqual(payload["metadata"]["saved_count"], 3)
        self.assertEqual(payload["metadata"]["saved_output_text_count"], 1)
        self.assertTrue(payload["cases"][0]["output_text_saved"])
        self.assertFalse(payload["cases"][1]["output_text_saved"])
        self.assertIsNone(payload["cases"][1]["output_text"])


if __name__ == "__main__":
    unittest.main()
