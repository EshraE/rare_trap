"""Optional helpers remain available without changing public paper workflows."""
import json

import numpy as np
import pytest
import torch

from raretrap import _projection, cli
from raretrap.configuration import (
    ESTIMATOR_PROJECTION_MODE, PAPER_PROJECTION_ARMS, default_experiment,
    experiments, validate_public_projection,
)
from scripts.run_projection_ablation import ARM_DEFINITIONS, _build_projection_arms, _prompt_for_arm
from utils.config import PROJECTION_SEED
from raretrap.terminology import IMPLEMENTATION_ARMS
from raretrap.projection import build_paper_projections


class Tokenizer:
    all_special_ids = []

    def __len__(self):
        return 30

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def build(arms):
    return _build_projection_arms(requested_arms=arms, tokenizer=Tokenizer(),
        embedding_weight=torch.randn(30, 8, generator=torch.Generator().manual_seed(9)),
        prompt_length=4, d_latent=3, seed=PROJECTION_SEED, covariance_eps=1e-5,
        lookup_device=torch.device("cpu"))


def test_every_internal_helper_remains_callable_without_changing_paper_maps():
    assert set(ARM_DEFINITIONS) == {"current_no_norm", "thinktrap_exact", "current_scaled",
                                    "random_projection", "vanilla_projection", "no_projection"}
    all_arms = build(list(ARM_DEFINITIONS))
    paper = build_paper_projections(tokenizer=Tokenizer(),
        embedding_weight=torch.randn(30, 8, generator=torch.Generator().manual_seed(9)),
        S=4, D=3, epsilon=1e-5, lookup_device=torch.device("cpu"))
    z = np.asarray([.2, -.3, .5], dtype=np.float32)
    for arm in all_arms.values():
        prompt = _prompt_for_arm(arm=arm, z=z, rng=np.random.default_rng(42))
        assert len(prompt.token_ids) == 4
        assert all(0 <= token < 30 for token in prompt.token_ids)
    for name in PAPER_PROJECTION_ARMS:
        first, second = all_arms[IMPLEMENTATION_ARMS[name]].projection, paper[name].projection
        for key in ("bias", "matrix", "shared_matrix", "slot_transforms", "token_ids"):
            a, b = getattr(first, key), getattr(second, key)
            assert (a is b is None) or (a is not None and b is not None and torch.equal(a, b))
        assert first.decode_z(z).token_ids == second.decode_z(z).token_ids


def test_dry_run_and_execution_share_the_same_two_arms(capsys, tmp_path):
    assert _projection.ARMS is PAPER_PROJECTION_ARMS
    assert PAPER_PROJECTION_ARMS == ("geometry_aware", "embedding_agnostic")
    cli.main(["projection", "--output", str(tmp_path / "new"), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["arms"] == list(_projection.ARMS)


def test_all_paper_presets_pass_the_public_projection_guard():
    for specification in experiments().values():
        validate_public_projection(specification["arguments"])
        assert specification["arguments"]["projection_mode"] == ESTIMATOR_PROJECTION_MODE


@pytest.mark.parametrize("mode", ["current_scaled", "random_projection", "vanilla_projection", "no_projection",
                                  "embedding-whitened", "thinktrap-exact"])
def test_modified_public_preset_cannot_select_an_internal_variant(monkeypatch, tmp_path, mode):
    spec = default_experiment("deepseek8b")
    spec["arguments"]["projection_mode"] = mode
    monkeypatch.setattr(cli, "preset", lambda name: spec)
    monkeypatch.setattr(cli, "_require_cuda", lambda: pytest.fail("must reject before loading"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--preset", "changed", "--output", str(tmp_path / "new"), "--dry-run"])
    assert exc.value.code == 2
    assert not (tmp_path / "new").exists()


def test_projection_seed_override_in_public_preset_is_rejected():
    spec = default_experiment("deepseek8b")
    spec["arguments"]["projection_seed"] = 5
    with pytest.raises(ValueError, match="internal"):
        validate_public_projection(spec["arguments"])


@pytest.mark.parametrize("command", ["projection", "estimate", "run"])
@pytest.mark.parametrize("option,value", [("--arms", "random_projection"), ("--projection-mode", "current_scaled")])
def test_public_cli_has_no_helper_selection_option(command, option, value, tmp_path):
    args = [command, "--output", str(tmp_path / "new"), option, value]
    if command == "run":
        args.extend(["--preset", "test"])
    with pytest.raises(SystemExit) as exc:
        cli.make_parser().parse_args(args)
    assert exc.value.code == 2
