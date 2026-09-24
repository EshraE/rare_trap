"""Command-line argument helpers for ThinkTrap runners."""

from __future__ import annotations

import argparse
import math

from utils.config import COND_PROB, PROJECTION_SEED


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    tested_llm_model_name: str,
    projection_mode: str,
    seed: int,
) -> None:
    parser.add_argument("--tested-llm-model", type=str, default=tested_llm_model_name)
    parser.add_argument(
        "--tested-model-revision",
        type=str,
        default=None,
        help="Pinned Hugging Face revision for the tested model and tokenizer.",
    )
    parser.add_argument(
        "--surrogate-token-model",
        type=str,
        default=None,
        help="Surrogate embedding/tokenizer model. Defaults to the tested model when omitted.",
    )
    parser.add_argument(
        "--surrogate-model-revision",
        type=str,
        default=None,
        help="Pinned Hugging Face revision for a distinct surrogate model and tokenizer.",
    )
    parser.add_argument("--input-prompt-len", type=int, default=20, help="Projected input prompt token slots")
    parser.add_argument("--llm-output-sampler", choices=("sample", "greedy"), default="greedy")
    parser.add_argument(
        "--chat-template-thinking",
        choices=("auto", "enabled", "disabled"),
        default="auto",
        help=(
            "Control tokenizer chat-template reasoning mode. 'enabled' passes "
            "enable_thinking=True, 'disabled' passes False, and 'auto' preserves the "
            "tokenizer default."
        ),
    )
    parser.add_argument(
        "--chat-template-reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help="Optional model-native reasoning effort passed to the chat template.",
    )
    parser.add_argument(
        "--require-thinking-template-effect",
        action="store_true",
        help=(
            "Fail before sampling unless enable_thinking=True and False render different "
            "tested-model prompt token IDs."
        ),
    )
    parser.add_argument(
        "--save-template-probe",
        type=str,
        default="",
        help="Optional JSON artifact proving the effective tested-model chat template.",
    )
    parser.add_argument("--llm-output-len-threshold", type=int, default=4096)
    parser.add_argument(
        "--performance-metric",
        choices=("output_length", "repetition_score"),
        default="output_length",
        help="Performance metric used by the subset estimator. Default preserves output-length runs.",
    )
    parser.add_argument(
        "--repetition-score-threshold",
        type=float,
        default=0.9,
        help="Failure threshold rho when --performance-metric=repetition_score.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=1,
        help="Number of prompts to evaluate per model.generate call; 1 preserves serial generation.",
    )
    parser.add_argument(
        "--ss-conditional-prob",
        type=float,
        default=COND_PROB,
        help="Subset-simulation conditional probability p0.",
    )
    parser.add_argument("--ss-samples-per-level", type=int, default=16, help="Subset-simulation samples per level")
    parser.add_argument("--ss-max-levels", type=int, default=4)
    parser.add_argument("--ss-sample-space-dim", type=int, default=24, help="Subset-simulation latent sample-space dimension")
    parser.add_argument(
        "--projection-mode",
        choices=("embedding-whitened", "factorized-embedding-whitened", "thinktrap-exact"),
        default=projection_mode,
    )
    parser.add_argument("--covariance-eps", type=float, default=1e-5)
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help=(
            f"Monte Carlo/MCMC and stochastic-decoding seed; defaults to {seed}. "
            f"The projection map uses the internal fixed seed {PROJECTION_SEED}."
        ),
    )
    parser.add_argument("--save-worst-cases", type=str, default="results/thinktrap_worst_cases.json")
    parser.add_argument(
        "--save-all-evaluations",
        type=str,
        default="",
        help="Optional JSON artifact containing every evaluated prompt and metric.",
    )
    parser.add_argument(
        "--save-raw-samples",
        type=str,
        default="",
        help=(
            "Optional append-only JSONL artifact containing the exact latent vector, "
            "prompt/model-input token IDs, completion token IDs, full completion text, "
            "and metrics for every model evaluation."
        ),
    )
    parser.add_argument(
        "--save-sampler-trace",
        type=str,
        default="",
        help=(
            "Optional directory for read-only sampler lineage, transition, state, "
            "population, and level artifacts."
        ),
    )
    parser.add_argument("--worst-case-count", type=int, default=100)
    parser.add_argument(
        "--case-output-text-count",
        type=int,
        default=100,
        help=(
            "Number of saved success-first top cases that retain full decoded model output text. "
            "All saved cases still retain decoded input prompts and token-length metadata."
        ),
    )
    parser.add_argument(
        "--objective-repeats",
        type=int,
        default=1,
        help="Number of stochastic generations per decoded prompt before aggregating the objective.",
    )
    parser.add_argument(
        "--objective-aggregation",
        choices=("mean", "median", "max", "p90", "top20mean"),
        default="mean",
        help="Aggregation over repeated output lengths for objective_repeats > 1.",
    )
    parser.add_argument("--sampler-progress", action="store_true")


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    conditional_prob = float(args.ss_conditional_prob)
    if not (0.0 < conditional_prob < 1.0):
        parser.error("--ss-conditional-prob must be in (0, 1).")
    min_samples_per_level = max(4, int(math.ceil(2.0 / conditional_prob)))
    if int(args.ss_samples_per_level) < min_samples_per_level:
        parser.error(
            f"--ss-samples-per-level must be at least {min_samples_per_level} when conditional_prob={conditional_prob}."
        )
    if int(args.llm_output_len_threshold) < 1:
        parser.error("--llm-output-len-threshold must be positive.")
    if int(args.input_prompt_len) < 1:
        parser.error("--input-prompt-len must be at least 1.")
    if int(args.ss_sample_space_dim) < 1:
        parser.error("--ss-sample-space-dim must be at least 1.")
    if int(args.ss_max_levels) < 0:
        parser.error("--ss-max-levels must be non-negative.")
    if float(args.covariance_eps) <= 0.0:
        parser.error("--covariance-eps must be positive.")
    if str(args.llm_output_sampler) == "sample" and int(args.ss_max_levels) > 0:
        parser.error(
            "Sample decoding is stochastic at fixed z and is only supported for direct Monte Carlo "
            "(--ss-max-levels 0). Use greedy decoding for subset simulation."
        )
    repetition_threshold = float(args.repetition_score_threshold)
    if not (0.0 <= repetition_threshold <= 1.0):
        parser.error("--repetition-score-threshold must be in [0, 1].")
    if int(args.generation_batch_size) < 1:
        parser.error("--generation-batch-size must be at least 1.")
    if int(args.worst_case_count) < 0:
        parser.error("--worst-case-count must be non-negative.")
    if int(args.case_output_text_count) < 0:
        parser.error("--case-output-text-count must be non-negative.")
    if int(args.objective_repeats) < 1:
        parser.error("--objective-repeats must be at least 1.")
