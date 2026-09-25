import hashlib
import json
import logging
from pathlib import Path
import signal
import threading
import time

import pytest

from raretrap.artifacts import DatasetJournal, atomic_json, dataset_rows, export_dataset
from raretrap.cli import main
from raretrap.lifecycle import RunInterrupted, RunMonitor, run_log


def sample(index=0):
    return dict(arm="geometry_aware", sample_index=index, prompt_text="hello", completion_text="world",
                output_tokens=2, repetition_score=.2, rep_2=0., rep_3=0., rep_4=0.,
                elapsed_sec=1., tokens_per_second=2., surrogate_token_ids=[1], model_input_token_ids=[1, 2],
                completion_token_ids=[3, 4], latent_vector_float32=[.1, -.2], success=False, hit_cap=False)


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_journal_incremental_partial_lines_and_digest(tmp_path):
    raw = tmp_path / "raw.jsonl"
    journal = DatasetJournal(raw, tmp_path / "dataset")
    try:
        assert journal.poll() == 0
        append(raw, {"record_type": "metadata"})
        append(raw, sample())
        incomplete = json.dumps(sample(1))
        with raw.open("a") as handle:
            handle.write(incomplete[:20])
        assert journal.poll() == 1
        assert journal.poll() == 0
        assert journal.manifest("running")["pending_raw_bytes"] == 20
        with raw.open("a") as handle:
            handle.write(incomplete[20:] + "\n")
        assert journal.poll() == 1
        state = journal.checkpoint("complete")
        payload = (tmp_path / "dataset/samples.jsonl").read_bytes()
        assert state["dataset_sha256"] == hashlib.sha256(payload).hexdigest()
        assert state["dataset_rows"] == state["completed_evaluations"] == 2
        assert state["pending_raw_bytes"] == 0
    finally:
        journal.close()


def test_estimator_dataset_has_one_row_per_repeat_and_no_metadata_row():
    generation = sample()
    raw = dict(record_type="sample", sample_index=7, subset_level=2, input_text="prompt",
               input_token_ids=[1, 9], latent_vector_float32=[.5], generations=[generation, generation])
    rows = list(dataset_rows(raw))
    assert len(rows) == 2
    assert rows[1]["repeat_index"] == 1
    assert rows[0]["surrogate_token_ids"] == [1, 9]
    assert rows[0]["model_input_token_ids"] == [1, 2]
    assert rows[0]["subset_level"] == 2
    assert rows[0]["arm"] is None
    assert list(dataset_rows({"record_type": "metadata"})) == []


def test_nonfinite_metric_marked_invalid_not_fabricated():
    row = sample()
    row["repetition_score"] = float("nan")
    exported = list(dataset_rows(row))[0]
    assert exported["invalid_metrics"]
    assert exported["metrics"]["repetition_score"] is None
    json.dumps(exported, allow_nan=False)


def test_corrupt_complete_line_is_rejected(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{bad json}\n')
    with pytest.raises(ValueError):
        export_dataset(raw, tmp_path / "dataset")
    assert json.loads((tmp_path / "dataset/manifest.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("payload", ["[]\n", "null\n", '"text"\n'])
def test_nonobject_raw_records_fail_cleanly(tmp_path, payload):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(payload)
    with pytest.raises(ValueError, match="must be objects"):
        export_dataset(raw, tmp_path / "dataset")
    assert json.loads((tmp_path / "dataset/manifest.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_snapshot_size_rejected_before_creating_destination(tmp_path, limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        DatasetJournal(tmp_path / "raw.jsonl", tmp_path / "dataset", byte_limit=limit)
    assert not (tmp_path / "dataset").exists()


def test_export_snapshot_does_not_modify_raw_or_overwrite(tmp_path):
    raw = tmp_path / "raw.jsonl"
    append(raw, sample())
    with raw.open("a") as handle:
        handle.write('{"sample')
    original = raw.read_bytes()
    manifest = export_dataset(raw, tmp_path / "export")
    assert manifest["completed_evaluations"] == 1
    assert manifest["pending_raw_bytes"] > 0
    assert manifest["state"] == "snapshot"
    assert raw.read_bytes() == original
    with pytest.raises(FileExistsError):
        export_dataset(raw, tmp_path / "export")


@pytest.mark.parametrize("change", ["removed", "replaced", "truncated"])
def test_journal_rejects_disrupted_raw_stream(tmp_path, change):
    raw = tmp_path / "raw.jsonl"
    append(raw, sample())
    journal = DatasetJournal(raw, tmp_path / "dataset")
    try:
        assert journal.poll() == 1
        if change == "removed":
            raw.unlink()
        elif change == "replaced":
            replacement = tmp_path / "replacement.jsonl"
            append(replacement, sample(1))
            replacement.replace(raw)
        else:
            raw.write_text("")
        with pytest.raises(ValueError, match="Raw stream"):
            journal.poll()
        with pytest.raises(ValueError, match="Raw stream"):
            journal.manifest("complete")
        assert journal.rows == journal.evaluations == 1
    finally:
        journal.close()


def test_journal_rejects_truncation_of_uncommitted_tail(tmp_path):
    raw = tmp_path / "raw.jsonl"
    append(raw, sample())
    committed = raw.read_bytes()
    with raw.open("ab") as stream:
        stream.write(b'{"pending":')
    journal = DatasetJournal(raw, tmp_path / "dataset")
    try:
        assert journal.poll() == 1
        raw.write_bytes(committed)
        with pytest.raises(ValueError, match="Raw stream"):
            journal.poll()
    finally:
        journal.close()


def test_snapshot_rejects_shortened_source_before_first_poll(tmp_path):
    raw = tmp_path / "raw.jsonl"
    append(raw, sample())
    append(raw, sample(1))
    journal = DatasetJournal(raw, tmp_path / "dataset", byte_limit=raw.stat().st_size)
    try:
        raw.write_text("")
        with pytest.raises(ValueError, match="Raw stream"):
            journal.poll()
    finally:
        journal.close()


def test_monitor_publishes_while_run_is_still_active(tmp_path):
    monitor = RunMonitor(tmp_path, interval=.01)
    monitor.start()
    try:
        append(tmp_path / "raw_samples.jsonl", sample())
        limit = time.monotonic() + 5
        while time.monotonic() < limit:
            status = json.loads((tmp_path / "status.json").read_text())
            if status["dataset_rows"] == 1:
                break
            time.sleep(.01)
        assert status["dataset_rows"] == 1
        assert status["state"] == "running"
    finally:
        monitor.finish("complete")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_complete"
    assert not monitor.thread.is_alive()


def test_run_log_final_drain_logging_and_redaction(monkeypatch, tmp_path):
    secret = "hf_" + "x" * 24
    monkeypatch.setenv("HF_TOKEN", secret)
    previous = list(logging.getLogger().handlers)
    with run_log(tmp_path):
        print("token=" + secret)
        logging.warning("diagnostic %s", secret)
        append(tmp_path / "raw_samples.jsonl", sample())
    assert logging.getLogger().handlers == previous
    log = (tmp_path / "run.log").read_text()
    assert secret not in log and "[REDACTED]" in log and "diagnostic" in log
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["state"] == "complete" and state["dataset_rows"] == 1


def test_sigterm_keeps_completed_records_and_restores_handler(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(RunInterrupted):
        with run_log(tmp_path):
            append(tmp_path / "raw_samples.jsonl", sample())
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == before
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "interrupted" and status["dataset_rows"] == 1
    assert (tmp_path / "FAILED").exists() and not (tmp_path / "DONE").exists()


def test_final_writer_failure_cannot_report_success(tmp_path):
    with pytest.raises(ValueError):
        with run_log(tmp_path):
            (tmp_path / "raw_samples.jsonl").write_text("invalid\n")
    assert (tmp_path / "FAILED").exists()
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"
    assert not any(t.name == "raretrap-artifacts" and t.is_alive() for t in threading.enumerate())


def test_status_reports_stale_not_assumed_alive(tmp_path, capsys):
    atomic_json(tmp_path / "status.json", dict(state="running", updated=time.time() - 100))
    main(["status", "--input", str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["stale"] is True


def test_export_cli(tmp_path, capsys):
    append(tmp_path / "raw_samples.jsonl", sample())
    main(["export-dataset", "--input", str(tmp_path), "--output", str(tmp_path / "export")])
    assert json.loads(capsys.readouterr().out)["dataset_rows"] == 1


def test_failed_marker_write_cannot_skip_monitor_cleanup(tmp_path, monkeypatch):
    original = Path.write_text
    def failing_marker(path, *args, **kwargs):
        if path.name == "FAILED":
            raise OSError("synthetic storage failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", failing_marker)
    with pytest.raises(RuntimeError, match="original failure"):
        with run_log(tmp_path):
            raise RuntimeError("original failure")
    assert not any(t.name == "raretrap-artifacts" and t.is_alive() for t in threading.enumerate())


def test_failed_atomic_update_keeps_previous_document(tmp_path):
    path = tmp_path / "status.json"
    atomic_json(path, {"value": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert json.loads(path.read_text()) == {"value": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_observation_does_not_consume_numerical_rng(tmp_path):
    import random
    import numpy as np
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    with run_log(tmp_path):
        append(tmp_path / "raw_samples.jsonl", sample())
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_background_failure_is_signalled_and_never_marked_complete(tmp_path, monkeypatch):
    import raretrap.lifecycle
    interrupted = threading.Event()
    monkeypatch.setattr(raretrap.lifecycle._thread, "interrupt_main", interrupted.set)
    monitor = RunMonitor(tmp_path, interval=.01)
    def disk_full():
        raise OSError("synthetic disk full")
    monkeypatch.setattr(monitor.journal, "poll", disk_full)
    monitor.start()
    assert interrupted.wait(timeout=5)
    with pytest.raises(RuntimeError, match="artifact writer failed"):
        monitor.finish("complete")
    assert not (tmp_path / "DONE").exists()


def test_projection_cli_full_dataset_with_synthetic_generation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    from raretrap import _projection, cli
    from utils.llm_generator import GenerationResult
    from utils.projection_space import ProjectedPrompt
    monkeypatch.setattr(cli, "_require_cuda", lambda: None)
    tokenizer = SimpleNamespace(all_special_ids=[])
    assets = SimpleNamespace(surrogate_tokenizer=tokenizer, tokenizer=tokenizer,
        surrogate_embedding_weight=torch.zeros(4, 3), device=torch.device("cpu"), dtype=torch.float16,
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768),
                              generation_config=SimpleNamespace(to_dict=lambda: {})))
    monkeypatch.setattr(_projection, "load_model_assets", lambda **kwargs: assets)
    p = SimpleNamespace(token_ids=torch.arange(4), embed_table=torch.zeros(4, 3), embed_norms=torch.zeros(4),
                        bias=torch.zeros(40, 3), matrix=None, shared_matrix=None, slot_transforms=None,
                        decode_z=lambda z: ProjectedPrompt(token_ids=[1] * 40, text="synthetic prompt"))
    monkeypatch.setattr(_projection, "build_paper_projections",
                        lambda **kwargs: {arm: SimpleNamespace(projection=p) for arm in _projection.ARMS})
    class Generator:
        def __init__(self, **kwargs):
            pass
        def _prompt_to_cpu_inputs(self, text):
            return {"input_ids": torch.tensor([1, 2])}
        def generate(self, text, **kwargs):
            return GenerationResult(prompt_text=text, input_token_count=2, completion_length=4,
                completion_text="test output", completion_token_ids=[3, 4, 5, 6], model_input_token_ids=[1, 2],
                elapsed_sec=1., tokens_per_second=4., hit_cap=False)
    monkeypatch.setattr(_projection, "HFTextGenerator", Generator)
    output = tmp_path / "run"
    main(["projection", "--samples", "2", "--output", str(output)])
    rows = [json.loads(line) for line in (output / "dataset/samples.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {row["arm"] for row in rows} == set(_projection.ARMS)
    assert rows[0]["latent_vector_float32"] == rows[2]["latent_vector_float32"]
    assert (output / "DONE").exists()
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
