"""The dedicated ablation runner is an alias, not a second implementation."""
import json
import sys

import pytest

from raretrap import cli
from scripts import run_paper_ablation


@pytest.mark.parametrize("model", [
    "deepseek8b", "gpt-oss20b", "mistral7b", "nemotron9b",
    "olmo3-7b", "phi4", "qwen3-14b", "qwen3.5-9b",
])
def test_ablation_dry_run_matches_public_workflow(model, tmp_path, capsys):
    output = tmp_path / "ablation"
    arguments = ["--model", model, "--output", str(output), "--dry-run"]
    run_paper_ablation.main(arguments)
    dedicated = json.loads(capsys.readouterr().out)
    cli.main(["projection", *arguments])
    assert dedicated == json.loads(capsys.readouterr().out)
    assert dedicated["arms"] == ["geometry_aware", "embedding_agnostic"]
    assert dedicated["samples"] == 100
    assert dedicated["cap"] == 20003
    assert dedicated["latent_dimension"] == 200
    assert dedicated["prompt_length"] == 40
    assert not output.exists()


def test_command_line_delegates_without_altering_arguments(monkeypatch):
    arguments = ["--model", "nemotron9b", "--output", "outputs/ablation"]
    monkeypatch.setattr(sys, "argv", ["raretrap-ablation", *arguments])
    calls = []
    monkeypatch.setattr(cli, "main", calls.append)
    run_paper_ablation.main()
    assert calls == [["projection", *arguments]]


@pytest.mark.parametrize("option,value", [
    ("--projection-seed", "5"), ("--arms", "current_scaled"),
])
def test_runner_does_not_expose_internal_overrides(option, value, tmp_path):
    with pytest.raises(SystemExit) as error:
        run_paper_ablation.main(["--output", str(tmp_path / "new"), option, value])
    assert error.value.code == 2
