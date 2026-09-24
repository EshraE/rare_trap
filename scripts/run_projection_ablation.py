#!/usr/bin/env python3
"""Generic latent projection ablation runner for local model comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from utils.objective_factory import (
    CANDIDATE_MODE,
    NOISE_K,
)
from utils.projection_space import (
    ProjectedPrompt,
    ProjectionSpace,
    _gaussian_block,
    _embedding_covariance_sqrt,
    _orthogonal_block,
    decode_token_ids,
    select_candidate_token_ids,
)


@dataclass
class Arm:
    name: str
    projection: ProjectionSpace | None
    token_ids: torch.Tensor
    tokenizer: object
    prompt_length: int


ARM_DEFINITIONS = {
    "current_no_norm": "mu + C^(1/2) Q B_s z",
    "current_scaled": "mu + sqrt(E/D) C^(1/2) Q B_s z",
    "random_projection": "mu + Q B_s z",
    "thinktrap_exact": "A_s z with independent A_ij ~ N(0, 1/D)",
    "vanilla_projection": "Q B_s z",
    "no_projection": "independent uniform draws from the filtered candidate IDs",
}


def _parse_arms(raw: str) -> list[str]:
    arms = [part.strip() for part in str(raw).split(",") if part.strip()]
    allowed = {
        "current_no_norm",
        "current_scaled",
        "random_projection",
        "thinktrap_exact",
        "vanilla_projection",
        "no_projection",
    }
    unknown = sorted(set(arms) - allowed)
    if unknown:
        raise ValueError(f"Unknown arms: {', '.join(unknown)}")
    if not arms:
        raise ValueError("At least one arm is required.")
    if len(set(arms)) != len(arms):
        raise ValueError("Ablation arm names must be unique.")
    return arms


def _run_metadata(
    *,
    complete: bool,
    started: str,
    args: argparse.Namespace,
    assets,
    arms: dict[str, Arm],
    samples: int,
    threshold: int,
    max_new_tokens: int,
    generation_batch_size: int,
    runtime: dict,
) -> dict:
    first_arm = next(iter(arms.values()))
    return {
        "artifact_schema_version": 2,
        "started": started,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete": bool(complete),
        "model": str(args.model),
        "surrogate": str(assets.surrogate_name),
        "tested_model_revision_requested": args.model_revision,
        "surrogate_model_revision_requested": (
            args.surrogate_revision if str(assets.surrogate_name) != str(args.model) else args.model_revision
        ),
        "tested_model_revision": assets.tested_model_revision,
        "tested_tokenizer_revision": assets.tested_tokenizer_revision,
        "surrogate_model_revision": assets.surrogate_model_revision,
        "surrogate_tokenizer_revision": assets.surrogate_tokenizer_revision,
        "samples": int(samples),
        "threshold": int(threshold),
        "max_new_tokens": int(max_new_tokens),
        "prompt_length": int(args.prompt_length),
        "latent_dim": int(args.latent_dim),
        "seed": int(args.seed),
        "arms": list(arms),
        "arm_projection_modes": {name: arm.name for name, arm in arms.items()},
        "arm_definitions": {name: ARM_DEFINITIONS[name] for name in arms},
        "paired_latent_samples": True,
        "candidate_filter": CANDIDATE_MODE,
        "candidate_count": int(first_arm.token_ids.numel()),
        "surrogate_embedding_dim": int(assets.surrogate_embedding_weight.shape[1]),
        "device": str(assets.device),
        "dtype": str(assets.dtype),
        "requested_generation_batch_size": int(args.generation_batch_size),
        "generation_batch_size": int(generation_batch_size),
        "chat_template_thinking": str(args.chat_template_thinking),
        "runtime": runtime,
    }


def _hash_text(text: str) -> str:
    return blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()


def _percentile(values: np.ndarray, q: float) -> float:
    if int(values.size) == 0:
        return 0.0
    return float(np.percentile(values, q))


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranks = np.empty(int(values.size), dtype=np.float64)
    if int(values.size) == 0:
        return ranks
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    start = 0
    while start < int(values.size):
        end = start + 1
        while end < int(values.size) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (float(start) + float(end - 1)) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if int(x.size) < 2 or int(y.size) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _score_distribution(records: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in records], dtype=np.float64)
    count = int(values.size)
    unique_count = int(np.unique(values).size) if count else 0
    return {
        f"min_{key}": float(np.min(values)) if count else 0.0,
        f"median_{key}": float(np.median(values)) if count else 0.0,
        f"p75_{key}": _percentile(values, 75),
        f"p90_{key}": _percentile(values, 90),
        f"p95_{key}": _percentile(values, 95),
        f"max_{key}": float(np.max(values)) if count else 0.0,
        f"unique_{key}_count": unique_count,
        f"unique_{key}_fraction": float(unique_count / count) if count else 0.0,
        f"{key}_tie_rate": 1.0 - float(unique_count / count) if count else 0.0,
        f"fraction_{key}_eq_0": float(np.mean(values == 0.0)) if count else 0.0,
    }


def _summary(records: list[dict], threshold: int) -> dict:
    lengths = np.asarray([int(row["output_tokens"]) for row in records], dtype=np.float64)
    repetition_scores = np.asarray(
        [float(row["repetition_score"]) for row in records],
        dtype=np.float64,
    )
    input_tokens = np.asarray([int(row["input_token_count"]) for row in records], dtype=np.float64)
    elapsed = np.asarray([float(row["elapsed_sec"]) for row in records], dtype=np.float64)
    success_count = int(np.sum(lengths >= int(threshold)))
    cap_count = int(sum(bool(row["hit_cap"]) for row in records))
    count = int(lengths.size)
    success_low, success_high = _wilson_interval(success_count, count)
    summary = {
        "count": count,
        "threshold": int(threshold),
        "success_count": success_count,
        "success_fraction": float(success_count / count) if count else 0.0,
        "success_fraction_wilson95_low": success_low,
        "success_fraction_wilson95_high": success_high,
        "cap_count": cap_count,
        "cap_fraction": float(cap_count / count) if count else 0.0,
        "min_output_tokens": int(np.min(lengths)) if count else 0,
        "mean_output_tokens": float(np.mean(lengths)) if count else 0.0,
        "median_output_tokens": float(np.median(lengths)) if count else 0.0,
        "std_output_tokens": float(np.std(lengths, ddof=1)) if count > 1 else 0.0,
        "p75_output_tokens": _percentile(lengths, 75),
        "p90_output_tokens": _percentile(lengths, 90),
        "p95_output_tokens": _percentile(lengths, 95),
        "p99_output_tokens": _percentile(lengths, 99),
        "max_output_tokens": int(np.max(lengths)) if count else 0,
        "mean_input_tokens": float(np.mean(input_tokens)) if count else 0.0,
        "median_input_tokens": float(np.median(input_tokens)) if count else 0.0,
        "mean_elapsed_sec": float(np.mean(elapsed)) if count else 0.0,
        "total_elapsed_sec": float(np.sum(elapsed)) if count else 0.0,
        "unique_prompt_count": len({row["prompt_hash"] for row in records}),
        "unique_output_count": len({row["output_hash"] for row in records}),
        "unique_success_prompt_count": len(
            {row["prompt_hash"] for row in records if int(row["output_tokens"]) >= int(threshold)}
        ),
        "unique_success_output_count": len(
            {row["output_hash"] for row in records if int(row["output_tokens"]) >= int(threshold)}
        ),
        "spearman_repetition_score_vs_output_tokens": _spearman(repetition_scores, lengths),
    }
    summary.update(_score_distribution(records, "repetition_score"))
    summary.update(_score_distribution(records, "rep_2"))
    summary.update(_score_distribution(records, "rep_3"))
    summary.update(_score_distribution(records, "rep_4"))
    return summary


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if int(total) <= 0:
        return 0.0, 0.0
    n = float(total)
    proportion = float(successes) / n
    denominator = 1.0 + (z * z) / n
    center = (proportion + (z * z) / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt((proportion * (1.0 - proportion) / n) + (z * z) / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _write_outputs(
    *,
    output_json: Path,
    output_csv: Path,
    payload: dict,
    records: list[dict],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_json.parent,
        prefix=f".{output_json.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json_text)
        temporary_json = Path(handle.name)
    temporary_json.replace(output_json)
    fieldnames = [
        "arm",
        "sample_index",
        "output_tokens",
        "repetition_score",
        "rep_2",
        "rep_3",
        "rep_4",
        "success",
        "hit_cap",
        "input_token_count",
        "elapsed_sec",
        "tokens_per_second",
        "prompt_hash",
        "output_hash",
        "surrogate_prompt_token_count",
        "prompt_char_count",
        "token_ids",
        "prompt_text",
    ]
    if any("completion_text" in row for row in records):
        fieldnames.append("completion_text")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_csv.parent,
        prefix=f".{output_csv.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
            doublequote=True,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(records)
        temporary_csv = Path(handle.name)
    temporary_csv.replace(output_csv)


def _build_projection_arms(
    *,
    requested_arms: Iterable[str],
    tokenizer,
    embedding_weight: torch.Tensor,
    prompt_length: int,
    d_latent: int,
    seed: int,
    covariance_eps: float,
    lookup_device: torch.device,
) -> dict[str, Arm]:
    requested_names = list(requested_arms)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    embed_weight = embedding_weight.detach().to("cpu", torch.float32)
    candidate_ids = select_candidate_token_ids(
        tokenizer=tokenizer,
        vocab_size=int(embed_weight.shape[0]),
        count=NOISE_K,
        rng=rng,
        mode=CANDIDATE_MODE,
    )
    token_ids = torch.tensor(candidate_ids, dtype=torch.long)
    embed_table = embed_weight.index_select(0, token_ids)
    embed_norms = torch.sum(embed_table * embed_table, dim=-1)
    mean, cov_sqrt = _embedding_covariance_sqrt(embed_table, eps=float(covariance_eps))

    if lookup_device.type == "cuda" and not torch.cuda.is_available():
        lookup_device = torch.device("cpu")
    if lookup_device.type == "mps":
        lookup_device = torch.device("cpu")

    slots = max(1, int(prompt_length))
    latent_dim = int(d_latent)
    embed_dim = int(embed_weight.shape[1])
    learned_bias = mean.unsqueeze(0).repeat(slots, 1).contiguous().to(lookup_device)
    zero_bias = torch.zeros((slots, embed_dim), dtype=torch.float32, device=lookup_device)
    q_shared = _orthogonal_block(embed_dim, latent_dim, rng)
    scale_correction = math.sqrt(float(embed_dim) / float(max(latent_dim, 1)))
    slot_transforms = torch.stack(
        [_orthogonal_block(latent_dim, latent_dim, rng) for _ in range(slots)],
        dim=0,
    ).contiguous()

    def make_projection(
        name: str,
        *,
        matrix: torch.Tensor | None = None,
        shared_matrix: torch.Tensor | None = None,
        bias: torch.Tensor,
    ) -> Arm:
        projection = ProjectionSpace(
            tokenizer=tokenizer,
            token_ids=token_ids.to(lookup_device),
            embed_table=embed_table.to(lookup_device),
            embed_norms=embed_norms.to(lookup_device),
            prompt_length=slots,
            d_latent=latent_dim,
            matrix=None if matrix is None else matrix.contiguous().to(lookup_device),
            bias=bias,
            mode=name,
            lookup_device=lookup_device,
            shared_matrix=None if shared_matrix is None else shared_matrix.contiguous().to(lookup_device),
            slot_transforms=None if shared_matrix is None else slot_transforms.to(lookup_device),
        )
        return Arm(
            name=name,
            projection=projection,
            token_ids=token_ids,
            tokenizer=tokenizer,
            prompt_length=slots,
        )

    arms: dict[str, Arm] = {}
    requested = set(requested_names)
    if "current_no_norm" in requested:
        arms["current_no_norm"] = make_projection(
            "factorized-embedding-whitened-no-norm",
            shared_matrix=cov_sqrt.matmul(q_shared),
            bias=learned_bias,
        )
    if "current_scaled" in requested:
        arms["current_scaled"] = make_projection(
            "factorized-embedding-whitened-scaled",
            shared_matrix=(scale_correction * cov_sqrt.matmul(q_shared)),
            bias=learned_bias,
        )
    if "random_projection" in requested:
        arms["random_projection"] = make_projection(
            "factorized-embedding-random-local",
            shared_matrix=q_shared,
            bias=learned_bias,
        )
    if "thinktrap_exact" in requested:
        arms["thinktrap_exact"] = make_projection(
            "thinktrap-exact",
            matrix=_gaussian_block(slots * embed_dim, latent_dim, rng),
            bias=zero_bias,
        )
    if "vanilla_projection" in requested:
        arms["vanilla_projection"] = make_projection(
            "factorized-orthogonal-gaussian",
            shared_matrix=q_shared,
            bias=zero_bias,
        )
    if "no_projection" in requested:
        arms["no_projection"] = Arm(
            name="uniform-random-token-local",
            projection=None,
            token_ids=token_ids,
            tokenizer=tokenizer,
            prompt_length=slots,
        )
    return {name: arms[name] for name in requested_names}


def _prompt_for_arm(
    *,
    arm: Arm,
    z: np.ndarray,
    rng: np.random.Generator,
) -> ProjectedPrompt:
    if arm.projection is not None:
        return arm.projection.decode_z(z)
    indices = rng.integers(0, int(arm.token_ids.numel()), size=int(arm.prompt_length))
    ids = [int(arm.token_ids[int(index)].item()) for index in indices]
    return ProjectedPrompt(token_ids=ids, text=decode_token_ids(arm.tokenizer, ids))
