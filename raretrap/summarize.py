"""Read generated artifacts without changing or pooling their estimands."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from raretrap.terminology import projection_name, projection_label
from raretrap.artifacts import jsonl_records


def _number(value: float, cap: int | None = None) -> str:
    if cap is not None and value == cap:
        return "cap"
    return f"{value:,.0f}" if cap is not None else f"{value:.3f}"


def _quantiles(values, percentiles=(50, 75, 95), cap=None):
    if not values:
        return "—"
    return " / ".join(_number(x, cap) for x in np.percentile(values, percentiles))


def projection_rows(root: Path, *, byte_limit: int | None = None) -> list[dict]:
    metadata = json.loads((root / "metadata.json").read_text())
    arms = [projection_name(name) for name in metadata["arms"]]
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("Unexpected or duplicate projection arms")
    expected = metadata["samples"]
    cap = metadata["max_new_tokens"]
    if type(expected) is not int or expected < 1 or type(cap) is not int or cap < 1:
        raise ValueError("Projection sample count and cap must be positive integers")
    records = defaultdict(list)
    for item in jsonl_records(root / "raw_samples.jsonl", committed_only=True, byte_limit=byte_limit):
        item["arm"] = projection_name(item["arm"])
        if item["arm"] not in arms:
            raise ValueError("Sample belongs to an unconfigured projection arm")
        index = item["sample_index"]
        if type(index) is not int or not 0 <= index < expected:
            raise ValueError("Projection sample index lies outside the recorded design")
        length, repetition = item["output_tokens"], item["repetition_score"]
        if type(length) is not int or not 0 <= length <= cap:
            raise ValueError("Invalid output length in projection artifact")
        if isinstance(repetition, bool) or not isinstance(repetition, (int, float)) or not 0 <= repetition <= 1:
            raise ValueError("Invalid repetition score in projection artifact")
        # Keep only statistics, not full completions and token arrays.
        records[item["arm"]].append({key: item[key] for key in
                                    ("sample_index", "output_tokens", "repetition_score")})
    rows = []
    for arm in arms:
        sample = records[arm]
        indices = [r["sample_index"] for r in sample]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate sample indices in {root.name}/{arm}")
        if len(sample) > expected:
            raise ValueError("More samples than the recorded design permits")
        rows.append(dict(model=metadata["model"], arm=arm, samples=f"{len(sample)}/{expected}",
            length=_quantiles([r["output_tokens"] for r in sample], cap=metadata["max_new_tokens"]),
            repetition=_quantiles([r["repetition_score"] for r in sample]),
            status="complete" if len(sample) == expected else "partial"))
    return rows


def population_quantiles(root: Path) -> dict[int, str]:
    slots = root / "sampler_trace/population_slots.jsonl"
    if not slots.exists():
        return {}
    levels = defaultdict(list)
    for row in jsonl_records(slots):
        levels[int(row["subset_level"])].append(float(row["performance_score"]))
    # All retained population occurrences count, including inherited/rejected states.
    return {level: " / ".join(f"{x:.6g}" for x in np.percentile(values, [25, 50, 75]))
            for level, values in sorted(levels.items())}


def is_fixed_budget_mc(metadata: dict) -> bool:
    """A completed IID budget does not require reaching an adaptive threshold."""
    result = metadata["estimator_result"]
    count = metadata.get("ss_samples_per_level")
    return (
        type(count) is int and count > 0
        and metadata.get("ss_max_levels") == 0
        and result.get("completed_levels") == 0
        and result.get("probability_method") == "direct_monte_carlo_full_level_zero"
        and result.get("stop_reason") in {"max_levels", "converged"}
        and result.get("total_evaluations") == result.get("final_population_size") == count
        and not any(item.get("invalid_evaluations", 0)
                    for item in metadata.get("threshold_probability_trace", []))
    )


def estimator_row(root: Path) -> dict:
    metadata = json.loads((root / "cases.json").read_text())["metadata"]
    result = metadata["estimator_result"]
    threshold = float(metadata["performance_threshold"])
    trajectory = [min(threshold, float(t["achieved_performance_threshold"]))
                  for t in metadata["threshold_probability_trace"]]
    return dict(model=metadata["tested_llm_model"], surrogate=metadata["surrogate_token_model"],
        metric=metadata["performance_metric"], threshold=threshold,
        probability=float(result["estimated_probability"]),
        evaluations=int(result["total_evaluations"]),
        levels=int(result["completed_levels"]) + 1,
        trajectory=" → ".join(f"{t:,.6g}" for t in trajectory),
        status="final" if is_fixed_budget_mc(metadata) else result["target_probability_status"],
        population_quantiles=population_quantiles(root))


def summarize(paths: list[Path]) -> None:
    projection, estimator = [], []
    for path in paths:
        if (path / "metadata.json").exists() and (path / "raw_samples.jsonl").exists():
            projection.extend(projection_rows(path))
        elif (path / "cases.json").exists():
            estimator.append(estimator_row(path))
        else:
            raise ValueError(f"No completed estimator or projection artifacts in {path}")
    print_tables(projection, estimator)


def print_tables(projection: list[dict], estimator: list[dict]) -> None:
    """Render already-read statistics without reopening growing run artifacts."""
    if projection:
        print("| Model | Projection | Samples | Length P50/P75/P95 | Repetition P50/P75/P95 | Status |")
        print("|---|---|---:|---|---|---|")
        for row in projection:
            arm = projection_label(row["arm"])
            print(f"| {row['model']} | {arm} | {row['samples']} | {row['length']} | {row['repetition']} | {row['status']} |")
    if estimator:
        print("| Target | Surrogate | Metric | Threshold | p-hat | Evals | Levels incl. 0 | Trajectory | Status |")
        print("|---|---|---|---:|---:|---:|---:|---|---|")
        for row in estimator:
            print(f"| {row['model']} | {row['surrogate']} | {row['metric']} | {row['threshold']:g} | "
                  f"{row['probability']:.6g} | {row['evaluations']:,} | {row['levels']} | {row['trajectory']} | {row['status']} |")
        print("\nRetained-population P25 / P50 / P75 (not proposal-stream quantiles):")
        for row in estimator:
            print(row["model"], row["metric"], row["population_quantiles"])
