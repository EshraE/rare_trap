import json

import pytest

from raretrap.summarize import estimator_row, is_fixed_budget_mc, population_quantiles, projection_rows


def test_partial_projection_is_not_reported_as_complete(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(dict(model="test", samples=100,
        arms=["geometry_aware", "embedding_agnostic"], max_new_tokens=20003)))
    row = dict(arm="geometry_aware", sample_index=0, output_tokens=20003, repetition_score=0.999999)
    (tmp_path / "raw_samples.jsonl").write_text(json.dumps(row) + "\n")
    rows = projection_rows(tmp_path)
    assert rows[0]["status"] == "partial"
    assert rows[0]["samples"] == "1/100"
    assert rows[0]["length"] == "cap / cap / cap"
    assert rows[1]["samples"] == "0/100"
    with (tmp_path / "raw_samples.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        projection_rows(tmp_path)


def test_population_summary_retains_multiplicities(tmp_path):
    trace = tmp_path / "sampler_trace"
    trace.mkdir()
    rows = [dict(subset_level=1, performance_score=x) for x in (10, 10, 10, 50)]
    (trace / "population_slots.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert population_quantiles(tmp_path)[1] == "10 / 10 / 20"


def mc_metadata(*, hits=0, count=1000):
    return dict(tested_llm_model="test", surrogate_token_model="test",
        performance_metric="output_length", performance_threshold=20000,
        ss_max_levels=0, ss_samples_per_level=count,
        threshold_probability_trace=[dict(achieved_performance_threshold=100,
                                         invalid_evaluations=0)],
        estimator_result=dict(estimated_probability=hits / count,
            total_evaluations=count, final_population_size=count, completed_levels=0,
            final_failure_count=hits, probability_method="direct_monte_carlo_full_level_zero",
            stop_reason="max_levels", target_reached=False,
            target_probability_status="nonfinal_diagnostic"))


@pytest.mark.parametrize("hits", [0, 1, 99])
def test_completed_fixed_budget_mc_is_final_even_with_few_hits(tmp_path, hits):
    metadata = mc_metadata(hits=hits)
    path = tmp_path / "cases.json"
    original = json.dumps(dict(metadata=metadata))
    path.write_text(original)
    row = estimator_row(tmp_path)
    assert row["status"] == "final"
    assert row["probability"] == hits / 1000
    assert row["levels"] == 1
    assert path.read_text() == original  # Preserve the original sampler record.


@pytest.mark.parametrize("change", [
    {"total_evaluations": 999}, {"final_population_size": 999},
    {"completed_levels": 1}, {"stop_reason": "invalid_evaluations"},
    {"stop_reason": "low_acceptance"},
    {"probability_method": "direct_monte_carlo_fixed_checkpoint"},
])
def test_non_iid_or_incomplete_runs_do_not_get_fixed_budget_finality(change):
    metadata = mc_metadata()
    metadata["estimator_result"].update(change)
    assert not is_fixed_budget_mc(metadata)


def test_conditional_and_invalid_populations_are_not_fixed_budget_mc():
    metadata = mc_metadata()
    metadata["ss_max_levels"] = 5
    assert not is_fixed_budget_mc(metadata)
    metadata["ss_max_levels"] = 0
    metadata["threshold_probability_trace"][0]["invalid_evaluations"] = 1
    assert not is_fixed_budget_mc(metadata)


def test_zero_hit_fixed_budget_mc_keeps_probability_and_gets_interval(tmp_path):
    pytest.importorskip("scipy")
    from raretrap.analysis import estimator_details

    (tmp_path / "cases.json").write_text(json.dumps(dict(metadata=mc_metadata())))
    row, _ = estimator_details(tmp_path)
    assert row["fixed_budget_mc"] and row["status"] == "final"
    assert row["probability"] == row["ci_lower"] == 0
    assert row["ci_upper"] == pytest.approx(1 - 0.025 ** (1 / 1000))
