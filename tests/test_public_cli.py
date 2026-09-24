import json
from collections import Counter

import pytest

from raretrap.cli import main, make_parser, resolve_experiment
from raretrap.configuration import experiments, models


def test_paper_defaults_and_no_projection_seed_option():
    parser = make_parser()
    args = parser.parse_args(["estimate", "--output", "outputs/test"])
    settings = resolve_experiment(args)["arguments"]
    assert (settings["input_prompt_len"], settings["ss_sample_space_dim"]) == (40, 200)
    assert (settings["ss_samples_per_level"], settings["ss_conditional_prob"], settings["ss_max_levels"]) == (1000, 0.1, 5)
    assert settings["llm_output_len_threshold"] == 20000
    assert settings["repetition_score_threshold"] == 0.99
    assert settings["projection_mode"] == "geometry_aware"
    for command in ("estimate", "projection", "run"):
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--output", "outputs/test", "--projection-seed", "999"])


def test_projection_defaults_and_latent_seed(capsys, tmp_path):
    main(["projection", "--output", str(tmp_path / "new"), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["cap"] == 20003
    assert result["samples"] == 100
    assert result["sampling_seed"] == 20260806
    assert result["arms"] == ["geometry_aware", "embedding_agnostic"]
    assert not (tmp_path / "new").exists()


def test_paper_sensitivity_settings_use_public_estimator_interface():
    args = make_parser().parse_args([
        "estimate", "--model", "qwen3-14b", "--surrogate", "self",
        "--metric", "repetition", "--prompt-length", "40", "--latent-dim", "100",
        "--seed", "20260917", "--max-levels", "5", "--output", "outputs/sensitivity",
    ])
    settings = resolve_experiment(args)["arguments"]
    assert settings["seed"] == 20260917
    assert (settings["input_prompt_len"], settings["ss_sample_space_dim"]) == (40, 100)
    assert (settings["ss_samples_per_level"], settings["ss_conditional_prob"],
            settings["ss_max_levels"]) == (1000, 0.1, 5)
    assert (settings["llm_output_len_threshold"], settings["repetition_score_threshold"]) == (20000, 0.99)
    assert (settings["performance_metric"], settings["projection_mode"]) == (
        "repetition_score", "geometry_aware")
    assert "projection_seed" not in settings


def test_refuses_existing_output(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["projection", "--output", str(tmp_path), "--dry-run"])
    assert exc.value.code == 2


def test_registry_contains_only_pinned_settings():
    registry = experiments()
    assert Counter(x["group"] for x in registry.values()) == dict(matched=32, additional=8, validation=30, mc=7, extreme=4)
    for entry in registry.values():
        args = entry["arguments"]
        assert "projection_seed" not in args
        assert args["llm_output_sampler"] == "greedy"
        assert args["projection_mode"] == "geometry_aware"
        assert args["covariance_eps"] == 1e-5
        for key in ("tested_model_revision", "surrogate_model_revision"):
            assert len(args[key]) == 40
            assert all(c in "0123456789abcdef" for c in args[key])
        assert entry["recorded_map_matches_release"] == (entry["group"] != "matched")
    assert len(models()) == 12


def test_every_preset_resolves_without_model_loading(tmp_path):
    parser = make_parser()
    for name in experiments():
        args = parser.parse_args(["run", "--preset", name, "--output", str(tmp_path / "new")])
        result = resolve_experiment(args)
        assert result["arguments"]["ss_samples_per_level"] >= 20


def test_runtime_metadata_does_not_collect_machine_identity():
    from utils.reproducibility import runtime_metadata
    result = runtime_metadata()
    assert not {"hostname", "working_directory", "python_executable", "command"} & result.keys()


def test_invalid_dry_run_fails_without_creating_output(tmp_path):
    for flag, value in (("--samples-per-level", "1"), ("--seed", "-1"), ("--max-levels", "-1")):
        with pytest.raises(SystemExit):
            main(["estimate", "--output", str(tmp_path / "new"), "--dry-run", flag, value])
    assert not (tmp_path / "new").exists()


def test_large_backend_batch_budget_matches_recorded_backends():
    for spec in experiments().values():
        if spec["group"] == "additional":
            expected = "32768" if spec["model"] == "nemotron3.5-30b" else "22048"
            assert spec["environment"]["THINKTRAP_VLLM_MAX_NUM_BATCHED_TOKENS"] == expected
