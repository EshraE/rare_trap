"""Reporting and worst-case file helpers for ThinkTrap experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from estimators_samplers import LevelSummary
from utils.config import PROJECTION_SEED
from utils.reproducibility import runtime_metadata
from utils.thinktrap_objective import ThinkTrapObjective


ARTIFACT_SCHEMA_VERSION = 5
ESTIMATOR_IMPLEMENTATION = "stable_order_statistic_fixed_level0_screen_v3"


def _final_population_exceedances_payload(result, obj) -> dict[str, object]:
    if result is None:
        return {
            "description": "No estimator result was supplied.",
            "selected_performance_metric": str(
                getattr(obj, "performance_metric", "output_length")
            ),
            "selected_performance_threshold": float(
                getattr(obj, "performance_threshold", getattr(obj, "threshold", 0.0))
            ),
            "final_population_size": 0,
            "exceedance_count": 0,
            "matched_metric_count": 0,
            "missing_metric_count": 0,
            "duplicate_retained_rows_preserved": True,
            "rows": [],
        }

    final_limit_state = np.asarray(getattr(result, "final_limit_state", []), dtype=np.float64)
    final_population = np.asarray(getattr(result, "final_population", []))
    exceedance_indices = np.flatnonzero(final_limit_state <= 0.0)
    rows: list[dict[str, object]] = []
    matched_count = 0
    lookup = getattr(obj, "metrics_for_latent_point", None)
    for index in exceedance_indices:
        metrics = None
        if callable(lookup) and final_population.ndim == 2 and int(index) < len(final_population):
            metrics = lookup(final_population[int(index)])
        row: dict[str, object] = {
            "final_population_index": int(index),
            "selected_limit_state": float(final_limit_state[int(index)]),
            "metrics_available": metrics is not None,
        }
        if metrics is not None:
            row.update(metrics)
            matched_count += 1
        rows.append(row)

    return {
        "description": (
            "One row per retained final-population sample meeting the selected threshold. "
            "Output length and repetition metrics come from the original model evaluation; "
            "copied or repeated retained states remain repeated rows."
        ),
        "selected_performance_metric": str(
            getattr(obj, "performance_metric", "output_length")
        ),
        "selected_performance_threshold": float(
            getattr(obj, "performance_threshold", getattr(obj, "threshold", 0.0))
        ),
        "final_population_size": int(final_limit_state.size),
        "exceedance_count": int(exceedance_indices.size),
        "matched_metric_count": int(matched_count),
        "missing_metric_count": int(exceedance_indices.size) - int(matched_count),
        "duplicate_retained_rows_preserved": True,
        "rows": rows,
    }


def _level_zero_screen_payload(result) -> dict[str, object] | None:
    screen = getattr(result, "level_zero_screen", None)
    if screen is None:
        return None
    return {
        "method": "fixed_level_zero_direct_monte_carlo_screen",
        "checkpoint_samples": int(screen.checkpoint_samples),
        "probability_boundary": float(screen.probability_boundary),
        "required_failures": int(screen.required_failures),
        "observed_failures": int(screen.observed_failures),
        "estimated_probability": float(screen.estimated_probability),
        "confidence_method": "wilson_score",
        "confidence_level": float(screen.confidence_level),
        "confidence_interval_lower": float(screen.confidence_interval_lower),
        "confidence_interval_upper": float(screen.confidence_interval_upper),
        "triggered": bool(screen.triggered),
        "decision": (
            "stop_and_skip_conditional_levels"
            if screen.triggered
            else "continue_full_level_zero"
        ),
    }


def _level_diversity_stats(obj, level: int) -> dict[str, float | int]:
    if hasattr(obj, "level_diversity_stats"):
        return obj.level_diversity_stats(int(level))
    return {
        "level_eval_count": 0,
        "level_unique_input_count": 0,
        "level_unique_output_count": 0,
        "level_unique_success_input_count": 0,
        "level_unique_success_output_count": 0,
        "level_input_duplicate_rate": 0.0,
        "level_output_duplicate_rate": 0.0,
        "level_top_output_repeats": 0,
    }


def _level_diversity_payload(obj) -> dict[str, dict[str, float | int]]:
    level_counts = getattr(obj, "level_eval_counts", {})
    return {
        str(level): _level_diversity_stats(obj, level)
        for level in sorted(level_counts.keys())
    }


def _level_sampler_payload(
    summaries,
    threshold: float,
    *,
    performance_metric: str = "output_length",
) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for summary in summaries or []:
        (
            min_score,
            mean_score,
            max_score,
            conditional_score_threshold,
        ) = _performance_summary(summary, threshold)
        proposed_moves = int(summary.proposed_moves)
        accepted_moves = int(summary.accepted_moves)
        population_size = int(summary.population_size)
        population_unique_points = int(summary.population_unique_points)
        item = {
            "level": int(summary.level),
            "evaluations_cumulative": int(summary.evaluations),
            "evaluations_this_level": (
                population_size if int(summary.level) == 0 else proposed_moves
            ),
            "population_size": population_size,
            "population_unique_z_count": population_unique_points,
            "population_duplicate_z_rate": _population_dup_z_rate(summary),
            "elite_count": int(summary.elite_count),
            "chain_length": int(summary.chain_length),
            "proposal_half_width": float(summary.proposal_half_width),
            "proposed_moves": proposed_moves,
            "accepted_moves": accepted_moves,
            "indicator_acceptance_rate": (
                float(summary.indicator_acceptance_rate)
                if summary.indicator_acceptance_rate is not None
                else None
            ),
            "rejected_moves": max(0, proposed_moves - accepted_moves),
            "rejection_rate": (
                1.0 - float(accepted_moves) / float(proposed_moves)
                if proposed_moves
                else None
            ),
            "n_at_or_above_level_threshold": int(summary.n_le_threshold),
            "empirical_conditional_threshold_fraction": (
                float(summary.n_le_threshold) / float(max(population_size, 1))
            ),
            "n_success": int(summary.n_fail),
            "probability_mass_before_level": float(
                summary.probability_mass_before_level
            ),
            "threshold_exceedance_probability": float(
                summary.threshold_exceedance_probability
            ),
            "threshold_probability_basis": (
                "fixed_checkpoint_empirical_fraction"
                if int(summary.elite_count) == 0
                else "nominal_subset_product"
            ),
            "target_exceedance_probability": float(
                summary.target_exceedance_probability
            ),
            "target_reached": bool(summary.target_reached),
            "performance_metric": str(performance_metric),
            "level_performance_threshold": float(conditional_score_threshold),
            "min_performance_score": float(min_score),
            "mean_performance_score": float(mean_score),
            "max_performance_score": float(max_score),
            "invalid_evaluations": int(summary.invalid_evaluations),
        }
        if str(performance_metric) == "output_length":
            item.update(
                {
                    "level_output_token_threshold": float(conditional_score_threshold),
                    "min_output_tokens": float(min_score),
                    "mean_output_tokens": float(mean_score),
                    "max_output_tokens": float(max_score),
                }
            )
        else:
            item.update(
                {
                    "level_repetition_score_threshold": float(conditional_score_threshold),
                    "min_repetition_score": float(min_score),
                    "mean_repetition_score": float(mean_score),
                    "max_repetition_score": float(max_score),
                }
            )
        payload[str(summary.level)] = item
    return payload


def _population_dup_z_rate(summary: LevelSummary) -> float:
    return 1.0 - (
        float(summary.population_unique_points) / float(max(summary.population_size, 1))
    )


def _chat_copy_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _performance_summary(summary: LevelSummary, threshold: float) -> tuple[float, float, float, float]:
    min_score = float(threshold) - float(summary.max_g)
    mean_score = float(threshold) - float(summary.mean_g)
    max_score = float(threshold) - float(summary.min_g)
    conditional_score_threshold = float(threshold) - float(summary.threshold)
    return (
        min_score,
        mean_score,
        max_score,
        conditional_score_threshold,
    )


def _threshold_probability_trace(
    summaries,
    threshold: float,
    *,
    performance_metric: str,
) -> list[dict[str, object]]:
    trace = []
    for summary in summaries or []:
        achieved_threshold = float(threshold) - float(summary.threshold)
        population_size = max(int(summary.population_size), 1)
        trace.append(
            {
                "level": int(summary.level),
                "performance_metric": str(performance_metric),
                "requested_performance_threshold": float(threshold),
                "achieved_performance_threshold": achieved_threshold,
                "limit_state_threshold": float(summary.threshold),
                "threshold_exceedance_probability": float(
                    summary.threshold_exceedance_probability
                ),
                "threshold_probability_basis": (
                    "fixed_checkpoint_empirical_fraction"
                    if int(summary.elite_count) == 0
                    else "nominal_subset_product"
                ),
                "probability_mass_before_level": float(
                    summary.probability_mass_before_level
                ),
                "nominal_conditional_probability": (
                    float(summary.elite_count) / float(population_size)
                    if int(summary.elite_count) > 0
                    else None
                ),
                "empirical_conditional_threshold_fraction": (
                    float(summary.n_le_threshold) / float(population_size)
                ),
                "target_exceedance_probability": float(
                    summary.target_exceedance_probability
                ),
                "target_reached": bool(summary.target_reached),
                "invalid_evaluations": int(summary.invalid_evaluations),
                "diagnostic_valid": int(summary.invalid_evaluations) == 0,
            }
        )
    return trace


def print_level_progress(
    label: str,
    summary: LevelSummary,
    obj: ThinkTrapObjective,
) -> None:
    evaluated_unique_input_count = len(obj.unique_input_hashes)
    evaluated_unique_success_input_count = len(obj.unique_success_input_hashes)
    evaluated_unique_z_count = len(obj.unique_z_hashes)
    performance_metric = str(getattr(obj, "performance_metric", "output_length"))
    performance_threshold = float(getattr(obj, "performance_threshold", getattr(obj, "threshold", 0)))
    (
        min_score,
        mean_score,
        max_score,
        conditional_score_threshold,
    ) = _performance_summary(summary, performance_threshold)

    parts = [f"[{label}] level={summary.level:02d}"]
    if performance_metric == "output_length":
        parts.extend(
            [
                f"level_output_token_threshold={conditional_score_threshold:.1f}",
                f"min_output_tokens={min_score:.1f}",
                f"mean_output_tokens={mean_score:.1f}",
                f"max_output_tokens={max_score:.1f}",
            ]
        )
    else:
        parts.extend(
            [
                f"level_repetition_score_threshold={conditional_score_threshold:.6f}",
                f"min_repetition_score={min_score:.6f}",
                f"mean_repetition_score={mean_score:.6f}",
                f"max_repetition_score={max_score:.6f}",
            ]
        )
    parts.append(
        f"threshold_probability={summary.threshold_exceedance_probability:.6e}"
    )
    parts.append(
        "threshold_probability_basis="
        + (
            "fixed_checkpoint_empirical_fraction"
            if summary.elite_count == 0
            else "nominal_subset_product"
        )
    )
    parts.append(
        f"target_probability_at_level={summary.target_exceedance_probability:.6e}"
    )
    parts.append(f"target_reached={str(summary.target_reached).lower()}")
    parts.append(f"evals={summary.evaluations}")
    if summary.level > 0:
        parts.append(f"w={summary.proposal_half_width:.4f}")
    if summary.proposed_moves > 0:
        accept_rate = summary.accepted_moves / max(summary.proposed_moves, 1)
        parts.append(f"accept={accept_rate:.4f}")
    if summary.n_fail > 0:
        parts.append(f"n_success={summary.n_fail}")
    parts.append(f"population_unique_z={summary.population_unique_points}")
    parts.append(f"evaluated_unique_inputs={evaluated_unique_input_count}")
    parts.append(f"evaluated_unique_z={evaluated_unique_z_count}")
    parts.append(f"evaluated_unique_success_inputs={evaluated_unique_success_input_count}")
    if summary.invalid_evaluations:
        parts.append(f"invalid={summary.invalid_evaluations}")
    print(" | ".join(parts), flush=True)

    diversity = _level_diversity_stats(obj, summary.level)
    print(
        f"[{label}-diversity] level={summary.level:02d} "
        f"population_dup_z_rate={_population_dup_z_rate(summary):.4f} "
        f"level_eval_count={diversity['level_eval_count']} "
        f"level_unique_inputs={diversity['level_unique_input_count']} "
        f"level_unique_outputs={diversity['level_unique_output_count']} "
        f"level_output_dup_rate={diversity['level_output_duplicate_rate']:.4f} "
        f"level_top_output_repeats={diversity['level_top_output_repeats']} "
        f"level_unique_success_outputs={diversity['level_unique_success_output_count']}",
        flush=True,
    )


def print_level_summaries(
    label: str,
    summaries,
    threshold: int,
    obj: ThinkTrapObjective | None = None,
) -> None:
    print(f"[{label}-levels] count={len(summaries)}")
    performance_metric = "output_length" if obj is None else str(getattr(obj, "performance_metric", "output_length"))
    performance_threshold = (
        float(threshold)
        if obj is None
        else float(getattr(obj, "performance_threshold", threshold))
    )
    for summary in summaries:
        (
            min_score,
            mean_score,
            max_score,
            conditional_score_threshold,
        ) = _performance_summary(summary, performance_threshold)
        parts = [
            f"[{label}-level] level={summary.level}",
            f"threshold_probability={summary.threshold_exceedance_probability:.6e}",
            "threshold_probability_basis="
            + (
                "fixed_checkpoint_empirical_fraction"
                if summary.elite_count == 0
                else "nominal_subset_product"
            ),
            f"target_probability_at_level={summary.target_exceedance_probability:.6e}",
            f"target_reached={str(summary.target_reached).lower()}",
            f"n_success={summary.n_fail}",
            f"n_at_or_above_level_threshold={summary.n_le_threshold}",
            f"population_unique_z={summary.population_unique_points}",
            f"evals={summary.evaluations}",
            f"elite={summary.elite_count}",
            f"chain={summary.chain_length}",
            f"w={summary.proposal_half_width:.4f}",
        ]
        if performance_metric == "output_length":
            parts[1:1] = [
                f"level_output_token_threshold={conditional_score_threshold:.1f}",
                f"min_output_tokens={min_score:.1f}",
                f"mean_output_tokens={mean_score:.1f}",
                f"max_output_tokens={max_score:.1f}",
            ]
        else:
            parts[1:1] = [
                f"level_repetition_score_threshold={conditional_score_threshold:.6f}",
                f"min_repetition_score={min_score:.6f}",
                f"mean_repetition_score={mean_score:.6f}",
                f"max_repetition_score={max_score:.6f}",
            ]
        if summary.indicator_acceptance_rate is not None:
            parts.append(f"indicator_accept={summary.indicator_acceptance_rate:.4f}")
        if summary.invalid_evaluations:
            parts.append(f"invalid={summary.invalid_evaluations}")
        print(" | ".join(parts))
        if obj is not None:
            diversity = _level_diversity_stats(obj, summary.level)
            print(
                f"[{label}-level-diversity] level={summary.level} "
                f"population_dup_z_rate={_population_dup_z_rate(summary):.4f} "
                f"level_eval_count={diversity['level_eval_count']} "
                f"level_unique_inputs={diversity['level_unique_input_count']} "
                f"level_unique_outputs={diversity['level_unique_output_count']} "
                f"level_output_dup_rate={diversity['level_output_duplicate_rate']:.4f} "
                f"level_top_output_repeats={diversity['level_top_output_repeats']} "
                f"level_unique_success_outputs={diversity['level_unique_success_output_count']}"
            )


def print_final_recap(label: str, result, threshold: int, obj: ThinkTrapObjective | None = None) -> None:
    performance_metric = "output_length" if obj is None else str(getattr(obj, "performance_metric", "output_length"))
    performance_threshold = (
        float(threshold)
        if obj is None
        else float(getattr(obj, "performance_threshold", threshold))
    )
    max_score = (
        float(performance_threshold) - float(np.min(result.final_limit_state))
        if int(result.final_limit_state.size)
        else float("nan")
    )
    score_label = "max_output_tokens" if performance_metric == "output_length" else "max_repetition_score"
    score_format = f"{max_score:.1f}" if performance_metric == "output_length" else f"{max_score:.6f}"
    terminal_summary = result.level_summaries[-1]
    achieved_threshold = performance_threshold - float(terminal_summary.threshold)
    achieved_label = (
        "achieved_output_token_threshold"
        if performance_metric == "output_length"
        else "achieved_repetition_score_threshold"
    )
    achieved_format = (
        f"{achieved_threshold:.1f}"
        if performance_metric == "output_length"
        else f"{achieved_threshold:.6f}"
    )
    target_probability_status = (
        "final" if bool(result.reliable_probability) else "nonfinal_diagnostic"
    )
    screen = getattr(result, "level_zero_screen", None)
    if screen is not None and screen.triggered:
        print(
            f"[{label}-final] stop={result.stop_reason.value} "
            "probability_method=direct_monte_carlo_fixed_checkpoint "
            f"levels={result.completed_levels} evals={result.total_evaluations} "
            f"successes={screen.observed_failures}/{screen.checkpoint_samples} "
            f"success_probability={result.estimated_probability:.6e} "
            f"ci95=[{screen.confidence_interval_lower:.6f},"
            f"{screen.confidence_interval_upper:.6f}] "
            f"screen_boundary={screen.probability_boundary:.6f} "
            f"target_reached={str(result.target_reached).lower()} "
            f"{achieved_label}={achieved_format} "
            f"achieved_threshold_probability={result.terminal_threshold_probability:.6e} "
            "achieved_threshold_probability_basis=fixed_checkpoint_empirical_fraction "
            f"target_probability_status={target_probability_status} "
            "conditional_subset_levels_skipped=true "
            f"reliable={result.reliable_probability} {score_label}={score_format}"
        )
    else:
        probability_method = str(
            getattr(
                result,
                "probability_method",
                (
                    "direct_monte_carlo_full_level_zero"
                    if int(result.completed_levels) == 0
                    else "subset_simulation"
                ),
            )
        )
        print(
            f"[{label}-final] stop={result.stop_reason.value} "
            f"probability_method={probability_method} "
            f"levels={result.completed_levels} evals={result.total_evaluations} "
            f"success_probability={result.estimated_probability:.6e} "
            f"target_probability_status={target_probability_status} "
            f"target_reached={str(result.target_reached).lower()} "
            f"{achieved_label}={achieved_format} "
            f"achieved_threshold_probability={result.terminal_threshold_probability:.6e} "
            "achieved_threshold_probability_basis=nominal_subset_product "
            f"reliable={result.reliable_probability} {score_label}={score_format}"
        )


def save_worst_cases(
    *,
    path: str,
    obj: ThinkTrapObjective,
    args: argparse.Namespace,
    case_count: int,
    result=None,
) -> None:
    if not path:
        return

    sorted_records = sorted(
        obj.best_records,
        key=lambda record: (
            -int(record.success),
            -float(record.performance_score if record.performance_score is not None else record.length),
            record.sample_index,
        ),
    )
    seen_inputs: set[str] = set()
    records = []
    for record in sorted_records:
        if record.rendered_input_text in seen_inputs:
            continue
        seen_inputs.add(record.rendered_input_text)
        records.append(record)
        if len(records) >= max(0, int(case_count)):
            break

    unique_input_text_count = len(obj.unique_input_hashes)
    unique_success_input_text_count = len(obj.unique_success_input_hashes)
    generator = getattr(obj, "generator", None)
    projection_space = getattr(obj, "projection_space", None)

    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "estimator_implementation": ESTIMATOR_IMPLEMENTATION,
        "runtime": runtime_metadata(getattr(generator, "device", None)),
        "tested_llm_model": args.tested_llm_model,
        "surrogate_token_model": args.surrogate_token_model,
        "tested_model_revision_requested": getattr(args, "tested_model_revision", None),
        "surrogate_model_revision_requested": getattr(args, "surrogate_model_revision", None),
        "input_mode": "chat",
        "llm_output_sampler": args.llm_output_sampler,
        "temperature": 1.0,
        "top_p": 1.0,
        "generation_batch_size": int(obj.generation_batch_size),
        "requested_generation_batch_size": int(getattr(args, "generation_batch_size", 1)),
        "objective_repeats": int(getattr(args, "objective_repeats", 1)),
        "objective_aggregation": str(getattr(args, "objective_aggregation", "mean")),
        "performance_metric": str(getattr(obj, "performance_metric", "output_length")),
        "performance_threshold": float(
            getattr(obj, "performance_threshold", int(args.llm_output_len_threshold))
        ),
        "repetition_score_threshold": float(
            getattr(args, "repetition_score_threshold", getattr(obj, "repetition_score_threshold", 0.9))
        ),
        "ss_samples_per_level": int(args.ss_samples_per_level),
        "ss_max_levels": int(args.ss_max_levels),
        "input_prompt_len": int(args.input_prompt_len),
        "ss_sample_space_dim": int(args.ss_sample_space_dim),
        "candidate_filter": "visible_nonspecial",
        "candidate_count": (
            int(projection_space.candidate_count) if projection_space is not None else None
        ),
        "surrogate_embedding_dim": (
            int(projection_space.embed_dim) if projection_space is not None else None
        ),
        "projection_mode": getattr(args, "projection_mode", "factorized-embedding-whitened"),
        "covariance_eps": float(getattr(args, "covariance_eps", 1e-5)),
        "max_new_tokens": int(
            getattr(generator, "max_new_tokens", int(args.llm_output_len_threshold) + 3)
        ),
        "llm_output_len_threshold": int(args.llm_output_len_threshold),
        "seed": int(args.seed),
        "record_count": int(obj.record_count),
        "unique_input_text_count": unique_input_text_count,
        "unique_output_text_count": len(getattr(obj, "unique_output_hashes", set())),
        "unique_success_input_text_count": unique_success_input_text_count,
        "unique_success_output_text_count": len(
            getattr(obj, "unique_success_output_hashes", set())
        ),
        "evaluated_unique_z_count": len(obj.unique_z_hashes),
        "level_diversity": _level_diversity_payload(obj),
        "level_sampler": _level_sampler_payload(
            getattr(result, "level_summaries", []),
            float(getattr(obj, "performance_threshold", int(args.llm_output_len_threshold))),
            performance_metric=str(getattr(obj, "performance_metric", "output_length")),
        ),
        "threshold_probability_trace": _threshold_probability_trace(
            getattr(result, "level_summaries", []) if result is not None else [],
            float(getattr(obj, "performance_threshold", int(args.llm_output_len_threshold))),
            performance_metric=str(getattr(obj, "performance_metric", "output_length")),
        ),
        "level_log_notes": {
            "level_diversity": (
                "Counts unique evaluated decoded prompts and generated outputs per level. "
                "These are proposal evaluations, not the retained MCMC population."
            ),
            "level_sampler": (
                "Counts subset/MCMC sampler state per level: retained population uniqueness, "
                "elite count, chain length, proposed moves, accepted indicator moves, and rejections."
            ),
            "level_zero": (
                "Level 0 is direct Monte Carlo. It has no MCMC proposed/accepted/rejected moves."
            ),
            "threshold_probability_trace": (
                "Pairs each achieved performance threshold with its cumulative exceedance "
                "probability. Intermediate threshold probabilities use the nominal fixed-elite "
                "subset product; empirical conditional fractions are also retained to expose ties."
            ),
        },
        "case_rank_policy": (
            "success_first_then_output_tokens"
            if str(getattr(obj, "performance_metric", "output_length")) == "output_length"
            else "success_first_then_selected_performance_score"
        ),
        "unique_case_policy": "exact_input_text",
        "stored_case_buffer_count": len(obj.best_records),
        "saved_count": len(records),
        "saved_success_count": sum(1 for record in records if record.success),
        "case_output_text_count": int(getattr(args, "case_output_text_count", 2)),
        "saved_output_text_count": sum(
            1 for record in records if record.completion_text is not None
        ),
    }
    metadata.update(dict(getattr(obj, "run_metadata", {})))
    final_population_exceedances = _final_population_exceedances_payload(result, obj)
    if result is not None:
        final_failure_count = int(np.count_nonzero(result.final_limit_state <= 0.0))
        level_zero_screen = _level_zero_screen_payload(result)
        metadata["estimator_result"] = {
            "stop_reason": result.stop_reason.value,
            "probability_method": str(
                getattr(
                    result,
                    "probability_method",
                    (
                        "direct_monte_carlo_fixed_checkpoint"
                        if level_zero_screen is not None and level_zero_screen["triggered"]
                        else (
                            "direct_monte_carlo_full_level_zero"
                            if int(result.completed_levels) == 0
                            else "subset_simulation"
                        )
                    ),
                )
            ),
            "reliable_probability": bool(result.reliable_probability),
            "estimated_probability": float(result.estimated_probability),
            "target_probability_status": (
                "final" if bool(result.reliable_probability) else "nonfinal_diagnostic"
            ),
            "target_reached": bool(getattr(result, "target_reached", False)),
            "terminal_limit_state_threshold": float(
                getattr(result, "terminal_threshold", float("nan"))
            ),
            "terminal_achieved_performance_threshold": (
                float(getattr(obj, "performance_threshold", int(args.llm_output_len_threshold)))
                - float(getattr(result, "terminal_threshold", float("nan")))
            ),
            "terminal_threshold_exceedance_probability": float(
                getattr(result, "terminal_threshold_probability", 0.0)
            ),
            "completed_levels": int(result.completed_levels),
            "total_evaluations": int(result.total_evaluations),
            "level_probabilities": [float(value) for value in result.level_probabilities],
            "final_population_size": int(result.final_limit_state.size),
            "final_failure_count": final_failure_count,
            "final_failure_fraction": (
                float(final_failure_count) / float(result.final_limit_state.size)
                if int(result.final_limit_state.size)
                else 0.0
            ),
            "final_exceedance_metrics_matched_count": int(
                final_population_exceedances["matched_metric_count"]
            ),
            "final_exceedance_metrics_missing_count": int(
                final_population_exceedances["missing_metric_count"]
            ),
            "level_zero_monte_carlo_screen": level_zero_screen,
        }
    payload = {
        "metadata": metadata,
        "final_population_exceedances": final_population_exceedances,
        "cases": [
            {
                "rank": index + 1,
                "sample_index": record.sample_index,
                "subset_level": record.subset_level,
                "output_tokens": record.length,
                "output_token_margin": record.output_token_margin,
                "performance_metric": record.performance_metric,
                "performance_score": record.performance_score,
                "performance_threshold": record.performance_threshold,
                "performance_margin": record.performance_margin,
                "repetition_score": record.repetition_score,
                "rep_2": record.rep_2,
                "rep_3": record.rep_3,
                "rep_4": record.rep_4,
                "success": record.success,
                "repeat_count": record.repeat_count,
                "repeat_aggregation": record.repeat_aggregation,
                "repeat_lengths": record.repeat_lengths,
                "best_repeat_index": record.best_repeat_index,
                "case_line": (
                    f"rank={index + 1} sample={record.sample_index} "
                    f"level={record.subset_level if record.subset_level is not None else 'unknown'} "
                    f"output_tokens={record.length} "
                    f"performance_score={float(record.performance_score if record.performance_score is not None else record.length):.6f} "
                    f"performance_margin={float(record.performance_margin if record.performance_margin is not None else record.output_token_margin):+.6f} "
                    f"success={str(record.success).lower()}"
                ),
                "input_text": record.rendered_input_text,
                "input_token_ids": record.input_token_ids,
                "input_text_chat_copy": _chat_copy_text(record.rendered_input_text),
                "input_text_chat_lines": _chat_copy_text(record.rendered_input_text).split("\n"),
                "output_text_saved": record.completion_text is not None,
                "output_text": record.completion_text,
            }
            for index, record in enumerate(records)
        ],
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(output_path)
    print(
        f"[top-cases] saved_unique_top={len(records)} "
        f"unique_inputs={unique_input_text_count} "
        f"unique_success_inputs={unique_success_input_text_count} "
        f"path={output_path}"
    )


def save_all_evaluations(
    *,
    path: str,
    obj: ThinkTrapObjective,
    args: argparse.Namespace,
    result=None,
) -> None:
    if not path:
        return

    records = list(getattr(obj, "evaluation_trace", []))
    payload = {
        "metadata": {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "description": (
                "One row per model evaluation. Conditional-level rows are MCMC proposals, "
                "not an unweighted sample from the original distribution."
            ),
            "tested_llm_model": args.tested_llm_model,
            "surrogate_token_model": args.surrogate_token_model,
            "tested_model_revision_requested": getattr(args, "tested_model_revision", None),
            "surrogate_model_revision_requested": getattr(args, "surrogate_model_revision", None),
            "estimator_seed": int(args.seed),
            "projection_seed": PROJECTION_SEED,
            "projection_mode": getattr(args, "projection_mode", None),
            "input_prompt_len": int(args.input_prompt_len),
            "ss_sample_space_dim": int(args.ss_sample_space_dim),
            "ss_samples_per_level": int(args.ss_samples_per_level),
            "ss_conditional_prob": float(args.ss_conditional_prob),
            "ss_max_levels": int(args.ss_max_levels),
            "llm_output_len_threshold": int(args.llm_output_len_threshold),
            "performance_metric": str(getattr(obj, "performance_metric", "output_length")),
            "record_count": len(records),
            "stop_reason": (
                getattr(getattr(result, "stop_reason", None), "value", None)
                if result is not None
                else None
            ),
        },
        "records": records,
    }
    payload["metadata"].update(dict(getattr(obj, "run_metadata", {})))
    payload["metadata"]["projection_seed"] = PROJECTION_SEED

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(output_path)
    print(f"[evaluations] saved={len(records)} path={output_path}")
