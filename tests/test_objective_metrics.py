from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from utils.llm_generator import GenerationResult
from utils.projection_space import ProjectedPrompt
from utils.thinktrap_objective import EvaluationRecord, ThinkTrapObjective


def _objective(metric: str = "repetition_score") -> ThinkTrapObjective:
    generator = SimpleNamespace(
        tokenizer=SimpleNamespace(all_special_ids=[0]),
        max_new_tokens=100,
    )
    projection = SimpleNamespace(d_latent=2)
    return ThinkTrapObjective(
        generator=generator,
        projection_space=projection,
        threshold=50,
        performance_metric=metric,
        repetition_score_threshold=0.8,
    )


def test_repetition_objective_uses_completion_tokens_and_selected_threshold() -> None:
    objective = _objective()
    result = GenerationResult(
        prompt_text="prompt",
        input_token_count=3,
        completion_length=5,
        completion_text="output",
        completion_token_ids=[0, 7, 7, 7, 0],
        elapsed_sec=1.0,
        tokens_per_second=5.0,
        hit_cap=False,
    )

    metrics = objective._metrics_from_result(
        result=result,
        elapsed_sec=1.0,
        tokens_per_second=5.0,
        hit_cap=False,
    )

    assert metrics.length == 5
    assert metrics.repetition_score == 0.5
    assert metrics.performance_score == 0.5
    assert metrics.performance_margin == pytest.approx(-0.3)
    assert not metrics.success


def test_saved_record_buffer_keeps_one_best_record_per_exact_prompt() -> None:
    objective = _objective(metric="output_length")

    def record(sample_index: int, length: int, text: str) -> EvaluationRecord:
        return EvaluationRecord(
            sample_index=sample_index,
            subset_level=0,
            length=length,
            output_token_margin=length - 50,
            success=length >= 50,
            hit_cap=False,
            elapsed_sec=1.0,
            tokens_per_second=float(length),
            input_token_count=2,
            rendered_input_text=text,
            completion_text=f"output-{sample_index}",
            performance_score=float(length),
        )

    objective._store_record(record(1, 60, "same"))
    objective._store_record(record(2, 80, "same"))
    objective._store_record(record(3, 70, "different"))

    assert [item.rendered_input_text for item in objective.best_records] == ["same", "different"]
    assert objective.best_records[0].length == 80


def test_latent_metric_lookup_matches_final_float64_population_point() -> None:
    objective = _objective(metric="repetition_score")
    result = GenerationResult(
        prompt_text="prompt",
        input_token_count=3,
        completion_length=5,
        completion_text="output",
        completion_token_ids=[7, 7, 7, 7, 7],
        elapsed_sec=1.0,
        tokens_per_second=5.0,
        hit_cap=False,
    )
    metrics = objective._metrics_from_result(
        result=result,
        elapsed_sec=1.0,
        tokens_per_second=5.0,
        hit_cap=False,
    )
    objective._store_latent_metrics(
        latent_point=np.asarray([0.123456789, -2.3456789], dtype=np.float32),
        sample_index=9,
        metrics=metrics,
    )

    stored = objective.metrics_for_latent_point(
        np.asarray([0.123456789, -2.3456789], dtype=np.float64)
    )

    assert stored is not None
    assert stored["source_sample_index"] == 9
    assert stored["output_tokens"] == 5
    assert stored["repetition_score"] == metrics.repetition_score


def test_evaluation_trace_keeps_prompt_metrics_and_hashes_without_output_text() -> None:
    objective = _objective(metric="output_length")
    objective.set_subset_level(2)
    result = GenerationResult(
        prompt_text="prompt",
        input_token_count=3,
        completion_length=55,
        completion_text="generated output",
        completion_token_ids=[7, 8, 9],
        elapsed_sec=1.0,
        tokens_per_second=55.0,
        hit_cap=False,
    )
    metrics = objective._metrics_from_result(
        result=result,
        elapsed_sec=1.0,
        tokens_per_second=55.0,
        hit_cap=False,
    )
    objective._record_evaluation(
        sample_index=4,
        prompt=ProjectedPrompt(token_ids=[11, 12], text="prompt"),
        result=result,
        metrics=metrics,
        latent_point=np.asarray([0.25, -0.5], dtype=np.float32),
    )

    row = objective.evaluation_trace[0]
    assert row["subset_level"] == 2
    assert row["output_tokens"] == 55
    assert row["input_token_ids"] == [11, 12]
    assert row["input_text"] == "prompt"
    assert len(row["input_hash"]) == 32
    assert len(row["output_hash"]) == 32
    assert len(row["latent_hash"]) == 32
    assert "output_text" not in row


def test_raw_sample_stream_is_lossless_and_append_only(tmp_path) -> None:
    raw_path = tmp_path / "raw_samples.jsonl"
    objective = _objective(metric="repetition_score")
    objective.raw_samples_path = str(raw_path)
    objective.initialize_raw_sample_stream({"estimator_seed": 77, "projection_seed": 1010})
    result = GenerationResult(
        prompt_text="prompt",
        input_token_count=3,
        completion_length=4,
        completion_text="generated output",
        completion_token_ids=[7, 8, 8, 9],
        elapsed_sec=1.0,
        tokens_per_second=4.0,
        hit_cap=False,
        model_input_token_ids=[1, 2, 3],
    )
    metrics = objective._metrics_from_result(
        result=result,
        elapsed_sec=1.0,
        tokens_per_second=4.0,
        hit_cap=False,
    )
    latent = np.asarray([0.25, -0.5], dtype=np.float32)
    objective._record_evaluation(
        sample_index=1,
        prompt=ProjectedPrompt(token_ids=[11, 12], text="prompt"),
        result=result,
        metrics=metrics,
        latent_point=latent,
    )

    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["record_type"] == "metadata"
    assert rows[1]["record_type"] == "sample"
    assert rows[1]["latent_vector_float32"] == [0.25, -0.5]
    assert rows[1]["input_token_ids"] == [11, 12]
    assert rows[1]["generations"][0]["model_input_token_ids"] == [1, 2, 3]
    assert rows[1]["generations"][0]["completion_token_ids"] == [7, 8, 8, 9]
    assert rows[1]["generations"][0]["completion_text"] == "generated output"

    with pytest.raises(FileExistsError):
        objective.initialize_raw_sample_stream({"estimator_seed": 88})
