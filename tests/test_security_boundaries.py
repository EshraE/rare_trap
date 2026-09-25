"""Operational safeguards must not change the scientific implementation."""
import importlib
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from raretrap import artifacts
from raretrap.analysis import export_cases, report
from raretrap.model_loading import local_model_code_only
from raretrap.summarize import projection_rows


@pytest.mark.parametrize("module_name", ["utils.model_assets", "raretrap.backends.model_assets"])
def test_target_and_surrogate_loads_explicitly_refuse_remote_code(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    calls = []

    def tokenizer(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(pad_token="pad", eos_token="end", _commit_hash=kwargs["revision"])

    class Model:
        def __init__(self, revision):
            self.config = SimpleNamespace(_commit_hash=revision)

        def to(self, *args):
            return self

        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(8, 4))

    def model(name, **kwargs):
        calls.append((name, kwargs))
        return Model(kwargs["revision"])

    original_tokenizer = SimpleNamespace(from_pretrained=tokenizer)
    original_model = SimpleNamespace(from_pretrained=model)
    monkeypatch.setattr(module, "AutoTokenizer", original_tokenizer)
    monkeypatch.setattr(module, "AutoModelForCausalLM", original_model)
    monkeypatch.setattr(module, "pick_device", lambda: torch.device("cpu"))
    monkeypatch.setenv("THINKTRAP_MODEL_DTYPE", "float32")
    if hasattr(module, "requested_generator_backend"):
        monkeypatch.setattr(module, "requested_generator_backend", lambda: "transformers")
        monkeypatch.setattr(module, "requested_model_device_map", lambda: None)
        monkeypatch.setattr(module, "requested_model_max_memory", lambda _: None)
    with local_model_code_only(module):
        loaded = module.load_model_assets(tested_llm_model_name="target",
            surrogate_token_model_name="surrogate", tested_model_revision="a" * 40,
            surrogate_model_revision="b" * 40)
    assert len(calls) == 4
    assert all(kwargs["trust_remote_code"] is False for _, kwargs in calls)
    assert [kwargs["revision"] for _, kwargs in calls] == ["a" * 40] * 2 + ["b" * 40] * 2
    assert loaded.dtype == torch.float32
    assert module.AutoTokenizer is original_tokenizer
    assert module.AutoModelForCausalLM is original_model


def test_loading_policy_restores_bindings_on_failure_without_rng_changes():
    def unexpected(*args, **kwargs):
        pytest.fail("Rejected remote-code request must not reach loader")
    loader = SimpleNamespace(from_pretrained=unexpected)
    module = SimpleNamespace(AutoTokenizer=loader, AutoModelForCausalLM=loader)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(ValueError, match="Remote model Python"):
        with local_model_code_only(module):
            module.AutoModelForCausalLM.from_pretrained("test", trust_remote_code=True)
    assert module.AutoTokenizer is module.AutoModelForCausalLM is loader
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def projection_fixture(root):
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps(dict(model="test", samples=2,
        arms=["geometry_aware", "embedding_agnostic"], max_new_tokens=20003)))
    row = dict(arm="geometry_aware", sample_index=0, output_tokens=10, repetition_score=.2,
               prompt_text="synthetic input", completion_text="synthetic output")
    (root / "raw_samples.jsonl").write_text(json.dumps(row) + "\n")
    return row


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission semantics")
def test_report_directory_is_private_even_with_permissive_umask(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "report"
    projection_fixture(source)
    previous = os.umask(0)
    try:
        report([source], destination, cases=True)
    finally:
        os.umask(previous)
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / "run_000/cases/case_000000/input.txt").read_text() == "synthetic input"


def test_summary_and_case_export_exclude_uncommitted_tail(tmp_path):
    source = tmp_path / "source"
    projection_fixture(source)
    with (source / "raw_samples.jsonl").open("ab") as stream:
        stream.write(b'{"arm":"geometry_aware","sample_index":1,"text":"\xe4')
    assert projection_rows(source)[0]["samples"] == "1/2"
    assert export_cases(source, tmp_path / "cases") == 1


@pytest.mark.parametrize("text", ['{broken}\n', '[]\n', 'null\n'])
def test_reader_rejects_malformed_complete_records(tmp_path, text):
    path = tmp_path / "raw.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(artifacts.jsonl_records(path, committed_only=True))


def test_reader_bounds_records_and_takes_a_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value":1}\n{"value":2}\n')
    reader = artifacts.jsonl_records(path)
    assert next(reader) == {"value": 1}
    with path.open("a") as stream:
        stream.write('{"value":3}\n')
    assert list(reader) == [{"value": 2}]
    monkeypatch.setattr(artifacts, "MAX_JSONL_LINE_BYTES", 4)
    with pytest.raises(ValueError, match="line size"):
        list(artifacts.jsonl_records(path))


def test_nonjournal_jsonl_allows_complete_final_record_without_newline(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value":1}')
    assert list(artifacts.jsonl_records(path)) == [{"value": 1}]
    assert list(artifacts.jsonl_records(path, committed_only=True)) == []


def test_reader_respects_supplied_snapshot_and_rejects_shorter_source(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value":1}\n')
    initial_size = path.stat().st_size
    with path.open("a") as stream:
        stream.write('{"value":2}\n')
    assert list(artifacts.jsonl_records(path, byte_limit=initial_size)) == [{"value": 1}]
    path.write_text("")
    with pytest.raises(ValueError, match="truncated"):
        list(artifacts.jsonl_records(path, byte_limit=initial_size))


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_reader_rejects_invalid_snapshot_sizes(tmp_path, limit):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value":1}\n')
    with pytest.raises(ValueError, match="nonnegative integer"):
        list(artifacts.jsonl_records(path, byte_limit=limit))


def test_reader_zero_byte_snapshot_stays_empty_after_append(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"value":1}\n')
    assert list(artifacts.jsonl_records(path, byte_limit=0)) == []


def test_recovery_checkpoint_failure_does_not_mask_original_error(tmp_path, monkeypatch):
    path = tmp_path / "raw.jsonl"
    path.write_text('{broken}\n')
    closed = []
    original_close = artifacts.DatasetJournal.close
    def close(journal):
        closed.append(True)
        original_close(journal)
    def no_space(*args):
        raise OSError("synthetic full disk")
    monkeypatch.setattr(artifacts.DatasetJournal, "checkpoint", no_space)
    monkeypatch.setattr(artifacts.DatasetJournal, "close", close)
    with pytest.raises(json.JSONDecodeError):
        artifacts.export_dataset(path, tmp_path / "dataset")
    assert closed == [True]
