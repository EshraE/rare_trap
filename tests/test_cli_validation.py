from __future__ import annotations

import argparse

import pytest

from utils.cli import add_common_arguments, validate_args


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_common_arguments(
        parser,
        tested_llm_model_name="target",
        projection_mode="factorized-embedding-whitened",
        seed=0,
    )
    return parser


def test_publication_defaults_are_deterministic_and_self_surrogate() -> None:
    parser = _parser()
    args = parser.parse_args([])

    assert args.llm_output_sampler == "greedy"
    assert args.surrogate_token_model is None
    assert args.tested_model_revision is None
    assert args.surrogate_model_revision is None
    assert args.projection_mode == "factorized-embedding-whitened"
    assert not hasattr(args, "projection_seed")
    assert args.save_all_evaluations == ""
    assert args.save_raw_samples == ""
    assert args.save_sampler_trace == ""


def test_projection_seed_is_not_a_cli_argument() -> None:
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--projection-seed", "99"])


def test_sample_decoding_is_restricted_to_direct_monte_carlo() -> None:
    parser = _parser()
    args = parser.parse_args(["--llm-output-sampler", "sample", "--ss-max-levels", "1"])

    with pytest.raises(SystemExit):
        validate_args(parser, args)

    direct_args = parser.parse_args(["--llm-output-sampler", "sample", "--ss-max-levels", "0"])
    validate_args(parser, direct_args)
