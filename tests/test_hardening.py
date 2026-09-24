import json
import subprocess

import pytest

from raretrap.cli import main
from raretrap.lifecycle import run_log
from raretrap.summarize import projection_rows


@pytest.mark.parametrize("command", ["estimate", "projection"])
@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_range_rejected_before_loading_models(tmp_path, command, seed):
    with pytest.raises(SystemExit) as exc:
        main([command, "--seed", str(seed), "--output", str(tmp_path / "new"), "--dry-run"])
    assert exc.value.code == 2
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("error", [RuntimeError("test failure"), KeyboardInterrupt()])
def test_failed_and_interrupted_runs_keep_log_but_no_done(tmp_path, error):
    with pytest.raises(type(error)):
        with run_log(tmp_path):
            print("started")
            raise error
    assert (tmp_path / "FAILED").read_text().strip() == type(error).__name__
    assert "started" in (tmp_path / "run.log").read_text()
    assert not (tmp_path / "DONE").exists()


@pytest.mark.parametrize("changes", [dict(arm="unknown"), dict(sample_index=-1), dict(sample_index=2),
    dict(sample_index=True), dict(repetition_score=float("nan")), dict(repetition_score=1.1),
    dict(output_tokens=20004), dict(output_tokens=-1)])
def test_corrupted_projection_rows_are_not_summarized(tmp_path, changes):
    metadata = dict(model="test", arms=["geometry_aware", "embedding_agnostic"], samples=2, max_new_tokens=20003)
    row = dict(arm="geometry_aware", sample_index=0, repetition_score=.5, output_tokens=10)
    row.update(changes)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "raw_samples.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        projection_rows(tmp_path)


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("git", 5)])
def test_git_metadata_is_optional(monkeypatch, tmp_path, error):
    import utils.reproducibility as metadata
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(metadata, "__file__", str(tmp_path / "utils/reproducibility.py"))

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(metadata.subprocess, "run", unavailable)
    assert metadata.git_state() == dict(commit=None, branch=None, dirty=None)


def test_installed_wheel_does_not_read_parent_repository(monkeypatch, tmp_path):
    import utils.reproducibility as metadata
    monkeypatch.setattr(metadata, "__file__", str(tmp_path / "site-packages/utils/reproducibility.py"))
    monkeypatch.setattr(metadata.subprocess, "run", lambda *a, **kw: pytest.fail("must not invoke git"))
    assert metadata.git_state() == dict(commit=None, branch=None, dirty=None)


def test_estimator_restores_factory_after_failure(monkeypatch, tmp_path):
    import raretrap.cli
    import raretrap._experiment
    import utils.objective_factory
    original = raretrap._experiment.build_obj
    monkeypatch.setattr(raretrap.cli, "_require_cuda", lambda: None)

    def failing_factory(**kwargs):
        raise RuntimeError("synthetic loading failure")

    monkeypatch.setattr(utils.objective_factory, "build_obj", failing_factory)
    with pytest.raises(SystemExit):
        main(["estimate", "--output", str(tmp_path / "run")])
    assert raretrap._experiment.build_obj is original
    assert (tmp_path / "run/FAILED").exists()
    assert not (tmp_path / "run/DONE").exists()


def test_projection_logs_loading_failure(monkeypatch, tmp_path):
    from raretrap import _projection

    def failing_loader(**kwargs):
        raise RuntimeError("synthetic loading failure")

    monkeypatch.setattr(_projection, "load_model_assets", failing_loader)
    with pytest.raises(RuntimeError):
        _projection.run_projection(model={"model": "test", "revision": "test"}, output=tmp_path / "run", samples=1)
    assert (tmp_path / "run/FAILED").exists()
    assert "synthetic loading failure" in (tmp_path / "run/run.log").read_text()
