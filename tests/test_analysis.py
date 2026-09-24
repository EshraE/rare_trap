import json

import pytest

from raretrap.analysis import clopper_pearson, estimator_details, evaluation_counts, latex_escape, report, replication_summaries


def projection_fixture(path):
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps(dict(model="test_model", samples=2,
        arms=["geometry_aware", "embedding_agnostic"], max_new_tokens=20003)))
    row = dict(arm="geometry_aware", sample_index=0, output_tokens=20003, repetition_score=1,
               prompt_text="test input", completion_text="test output", surrogate_token_ids=[7],
               completion_token_ids=[8], latent_vector_float32=[0.5])
    (path / "raw_samples.jsonl").write_text(json.dumps(row) + "\n")


def test_report_preserves_partial_status_and_exports_complete_records(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "report"
    projection_fixture(source)
    report([source], destination, cases=True)
    data = json.loads((destination / "projection.json").read_text())
    assert data[0]["samples"] == "1/2"
    assert data[0]["status"] == "partial"
    assert data[1]["samples"] == "0/2"
    assert "partial" in (destination / "projection.tex").read_text()
    case = destination / "run_000/cases/case_000000"
    assert (case / "input.txt").read_text() == "test input"
    assert (case / "output_000.txt").read_text() == "test output"
    assert json.loads((case / "record.json").read_text())["completion_token_ids"] == [8]
    with pytest.raises(FileExistsError):
        report([source], destination)
    with pytest.raises(ValueError, match="Duplicate"):
        report([source, source], tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()


def test_cp_zero_and_all_hits():
    pytest.importorskip("scipy")
    low, high = clopper_pearson(0, 100)
    assert low == 0
    assert high == pytest.approx(1 - 0.025 ** (1 / 100))
    low, high = clopper_pearson(100, 100)
    assert high == 1
    assert low == pytest.approx(0.025 ** (1 / 100))


def test_model_evaluation_counts_are_not_proposal_counts():
    history = {
        "1": dict(level=1, evaluations_cumulative=17, evaluations_this_level=8,
                  proposed_moves=8),
        "0": dict(level=0, evaluations_cumulative=10, evaluations_this_level=10),
    }
    counts = evaluation_counts(history)
    assert [row["model_evaluations"] for row in counts] == [10, 7]
    assert history["1"]["evaluations_this_level"] == 8
    assert evaluation_counts(None) == []
    history["1"]["evaluations_cumulative"] = 9
    with pytest.raises(ValueError, match="nondecreasing"):
        evaluation_counts(history)


def test_no_binomial_interval_for_checkpoint_selected_or_mcmc_data(tmp_path):
    metadata = dict(tested_llm_model="test", surrogate_token_model="test", performance_metric="output_length",
                    performance_threshold=20000, threshold_probability_trace=[], ss_max_levels=5,
                    estimator_result=dict(estimated_probability=.2, total_evaluations=200, completed_levels=0,
                        target_probability_status="final", probability_method="direct_monte_carlo_fixed_checkpoint",
                        target_reached=True, stop_reason="direct_monte_carlo_screen", final_population_size=200,
                        final_failure_count=40))
    (tmp_path / "cases.json").write_text(json.dumps(dict(metadata=metadata)))
    row, _ = estimator_details(tmp_path)
    assert row["ci_lower"] is None
    assert not row["fixed_budget_mc"]


def test_latex_values_are_escaped():
    assert latex_escape("a_b&c%") == r"a\_b\&c\%"


def test_replications_ignore_timestamps_but_reject_duplicate_seeds():
    metadata = dict(runtime={"recorded_at_utc": "first", "packages": {"numpy": "test"}},
                    estimator_result={"level_probabilities": [.1, .1]})
    row = dict(run="a", seed=1, probability=.01, method="subset_simulation", status="final")
    other_metadata = dict(metadata, runtime=dict(metadata["runtime"], recorded_at_utc="second"))
    other = dict(row, run="b", seed=2, probability=.02)
    result = replication_summaries([(row, metadata), (other, other_metadata)])
    assert len(result) == 1
    assert result[0]["mean"] == pytest.approx(.015)
    assert result[0]["sample_sd"] == pytest.approx(.01 / 2 ** .5)
    other["seed"] = 1
    result = replication_summaries([(row, metadata), (other, other_metadata)])
    assert result[0]["mean"] is None
    assert result[0]["sample_sd"] is None


def test_figures_are_generated_without_model_loading(tmp_path):
    pytest.importorskip("matplotlib")
    source, destination = tmp_path / "source", tmp_path / "report"
    projection_fixture(source)
    report([source], destination, figures=True)
    assert (destination / "run_000/projection_ecdf.svg").stat().st_size > 0
