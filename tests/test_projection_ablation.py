from __future__ import annotations

import pytest

from scripts.run_projection_ablation import _parse_arms, _summary


def test_arm_parser_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        _parse_arms("current_no_norm,current_no_norm")


def test_summary_reports_repetition_and_binomial_interval() -> None:
    records = [
        {
            "output_tokens": length,
            "repetition_score": repetition,
            "rep_2": repetition,
            "rep_3": repetition,
            "rep_4": repetition,
            "input_token_count": 10,
            "elapsed_sec": 1.0,
            "hit_cap": False,
            "prompt_hash": f"prompt-{index}",
            "output_hash": f"output-{index}",
        }
        for index, (length, repetition) in enumerate([(10, 0.1), (30, 0.9)])
    ]

    summary = _summary(records, threshold=20)

    assert summary["success_fraction"] == 0.5
    assert 0.0 < summary["success_fraction_wilson95_low"] < 0.5
    assert 0.5 < summary["success_fraction_wilson95_high"] < 1.0
    assert summary["median_repetition_score"] == 0.5
