"""Paper terminology changes labels, not the numerical experiment."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from raretrap.configuration import default_experiment, implementation_settings
from raretrap.projection import build_paper_projections
from raretrap.summarize import projection_rows, summarize
from raretrap.analysis import report
from raretrap.terminology import (
    IMPLEMENTATION_ESTIMATOR_MODE, IMPLEMENTATION_ARMS, PAPER_PROJECTION_ARMS,
    projection_name,
)
from scripts.run_projection_ablation import _build_projection_arms


class Tokenizer:
    all_special_ids = []

    def __len__(self):
        return 30

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def test_paper_projection_tensor_and_token_identity():
    W = torch.randn(30, 8, generator=torch.Generator().manual_seed(9))
    arguments = dict(tokenizer=Tokenizer(), embedding_weight=W, lookup_device=torch.device("cpu"))
    original = _build_projection_arms(requested_arms=list(IMPLEMENTATION_ARMS.values()),
        prompt_length=4, d_latent=3, seed=1010, covariance_eps=1e-5, **arguments)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    paper = build_paper_projections(S=4, D=3, epsilon=1e-5, **arguments)
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert tuple(paper) == PAPER_PROJECTION_ARMS
    Z = np.random.default_rng(42).standard_normal((20, 3)).astype(np.float32)
    for name, arm in paper.items():
        a, b = arm.projection, original[IMPLEMENTATION_ARMS[name]].projection
        for field in ("bias", "matrix", "shared_matrix", "slot_transforms", "token_ids", "embed_table", "embed_norms"):
            x, y = getattr(a, field), getattr(b, field)
            assert (x is y is None) or (x is not None and y is not None and torch.equal(x, y))
        for z in Z:
            assert a.decode_z(z) == b.decode_z(z)


def test_estimator_translation_changes_only_label_without_mutating_public_settings():
    public = default_experiment("deepseek8b")["arguments"]
    before = copy.deepcopy(public)
    numerical = implementation_settings(public)
    assert public == before and public["projection_mode"] == "geometry_aware"
    assert numerical.pop("projection_mode") == IMPLEMENTATION_ESTIMATOR_MODE
    assert numerical == {k: v for k, v in before.items() if k != "projection_mode"}


def write_projection(root, names):
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps(dict(model="test", samples=1,
        arms=names, max_new_tokens=20003)))
    (root / "raw_samples.jsonl").write_text("".join(json.dumps(dict(arm=name,
        sample_index=0, output_tokens=20003, repetition_score=1.0)) + "\n" for name in names))


def test_new_artifacts_produce_paper_reports_without_modifying_records(tmp_path, capsys):
    new = tmp_path / "new"
    write_projection(new, list(PAPER_PROJECTION_ARMS))
    original = (new / "raw_samples.jsonl").read_bytes()
    assert [row["arm"] for row in projection_rows(new)] == list(PAPER_PROJECTION_ARMS)
    summarize([new])
    text = capsys.readouterr().out
    assert "Geometry-aware" in text and "Embedding-agnostic" in text
    report([new], tmp_path / "report")
    latex = (tmp_path / "report/projection.tex").read_text()
    assert "Geometry-aware" in latex and "Embedding-agnostic" in latex
    assert all(name not in latex for name in IMPLEMENTATION_ARMS.values())
    assert (new / "raw_samples.jsonl").read_bytes() == original


def test_duplicate_arm_cannot_be_counted_twice(tmp_path):
    root = tmp_path / "run"
    write_projection(root, ["geometry_aware", "geometry_aware"])
    with pytest.raises(ValueError, match="duplicate"):
        projection_rows(root)


@pytest.mark.parametrize("name", ["current_no_norm", "thinktrap_exact", "current_scaled", "random_projection", "unknown"])
def test_readers_accept_only_paper_identifiers(name, tmp_path):
    from raretrap.artifacts import dataset_rows
    with pytest.raises(ValueError):
        projection_name(name)
    root = tmp_path / "run"
    write_projection(root, [name])
    with pytest.raises(ValueError):
        projection_rows(root)
    with pytest.raises(ValueError):
        list(dataset_rows(dict(arm=name)))


def test_estimator_records_paper_name_and_translates_only_at_factory(monkeypatch, tmp_path):
    from raretrap import cli, _experiment
    import utils.objective_factory
    calls = []
    expected = object()
    def factory(**kwargs):
        calls.append(kwargs)
        return expected
    def run(args):
        assert args.projection_mode == "geometry_aware"
        assert _experiment.build_obj(projection_mode=args.projection_mode) is expected
    monkeypatch.setattr(cli, "_require_cuda", lambda: None)
    monkeypatch.setattr(utils.objective_factory, "build_obj", factory)
    monkeypatch.setattr(_experiment, "run_experiment", run)
    previous = _experiment.build_obj
    cli.main(["estimate", "--output", str(tmp_path / "run")])
    assert calls == [{"projection_mode": IMPLEMENTATION_ESTIMATOR_MODE}]
    assert _experiment.build_obj is previous
    spec = json.loads((tmp_path / "run/configuration.json").read_text())
    assert spec["arguments"]["projection_mode"] == "geometry_aware"
