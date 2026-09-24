"""Persist and enrich read-only subset trace observations."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from estimators_samplers.trace_observer import SubsetTraceObserver
from utils.config import PROJECTION_SEED
from utils.reproducibility import runtime_metadata
from utils.thinktrap_objective import ThinkTrapObjective


TRACE_ARTIFACT_SCHEMA_VERSION = 1


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_record(summary: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(summary):
        return dataclasses.asdict(summary)
    if hasattr(summary, "__dict__"):
        return dict(vars(summary))
    raise TypeError(f"Unsupported level summary type: {type(summary)!r}")


def save_subset_trace_checkpoint(
    *,
    path: str,
    observer: SubsetTraceObserver | None,
    completed_level_summaries: Iterable[Any],
) -> None:
    """Atomically preserve raw observer state after a completed sampler level."""
    if not path or observer is None:
        return
    output_dir = Path(path) / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_ids = np.asarray([row["state_id"] for row in observer.states], dtype="U16")
    vector_path = output_dir / "latent_states.npz"
    vector_temporary = output_dir / "latent_states.tmp.npz"
    np.savez_compressed(
        vector_temporary,
        state_ids=state_ids,
        latent_vectors=observer.state_vectors,
    )
    vector_temporary.replace(vector_path)

    _write_jsonl(output_dir / "states.jsonl", observer.states)
    _write_jsonl(output_dir / "transitions.jsonl", observer.transitions)
    _write_jsonl(output_dir / "population_slots.jsonl", observer.population_slots)
    _write_json(
        output_dir / "levels.json",
        {
            "sampler_level_summaries": [
                _summary_record(summary) for summary in completed_level_summaries
            ],
            "observer_population_summaries": list(observer.population_summaries),
        },
    )
    _write_json(
        output_dir / "checkpoint.json",
        {
            "artifact_schema_version": TRACE_ARTIFACT_SCHEMA_VERSION,
            "complete": False,
            "completed_level_count": len(observer.population_summaries),
            "highest_completed_subset_level": (
                int(observer.population_summaries[-1]["subset_level"])
                if observer.population_summaries
                else None
            ),
            "record_counts": {
                "observed_evaluations": observer.observed_evaluation_count,
                "states": len(observer.states),
                "transitions": len(observer.transitions),
                "population_slots": len(observer.population_slots),
            },
        },
    )


def _distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _prompt_token_distance(left: list[int], right: list[int]) -> tuple[int, float]:
    width = max(len(left), len(right))
    if width == 0:
        return 0, 0.0
    mismatches = sum(
        1
        for index in range(width)
        if (left[index] if index < len(left) else None)
        != (right[index] if index < len(right) else None)
    )
    return mismatches, float(mismatches) / float(width)


def _enrich_trace(
    observer: SubsetTraceObserver,
    obj: ThinkTrapObjective,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_evaluations = [dict(row) for row in getattr(obj, "evaluation_trace", [])]
    if len(raw_evaluations) != observer.observed_evaluation_count:
        raise ValueError(
            "Observer/evaluation count mismatch: "
            f"observer={observer.observed_evaluation_count} "
            f"evaluations={len(raw_evaluations)}"
        )
    states = [dict(row) for row in observer.states]
    transitions = [dict(row) for row in observer.transitions]
    population_slots = [dict(row) for row in observer.population_slots]
    state_by_id = {str(row["state_id"]): row for row in states}
    state_by_hash = {str(row["latent_hash"]): row for row in states}

    first_evaluation_by_hash: dict[str, int] = {}
    for ordinal, row in enumerate(raw_evaluations):
        point_hash = row.get("latent_hash")
        if point_hash is not None:
            first_evaluation_by_hash.setdefault(str(point_hash), ordinal)
    for state in states:
        if state.get("source_evaluation_ordinal") is None:
            state["source_evaluation_ordinal"] = first_evaluation_by_hash.get(
                str(state["latent_hash"])
            )

    transition_by_evaluation = {
        int(row["evaluation_ordinal"]): row
        for row in transitions
        if row.get("evaluation_ordinal") is not None
    }
    for ordinal, transition in transition_by_evaluation.items():
        if ordinal >= len(raw_evaluations):
            raise ValueError(f"Transition references missing evaluation ordinal {ordinal}.")
        evaluation = raw_evaluations[ordinal]
        if str(evaluation.get("latent_hash")) != str(transition["proposal_latent_hash"]):
            raise ValueError(
                "Observer/evaluation latent mismatch at ordinal "
                f"{ordinal}: observer={transition['proposal_latent_hash']} "
                f"evaluation={evaluation.get('latent_hash')}"
            )
        if int(evaluation.get("subset_level", 0) or 0) != int(transition["subset_level"]):
            raise ValueError(
                "Observer/evaluation level mismatch at ordinal "
                f"{ordinal}: observer={transition['subset_level']} "
                f"evaluation={evaluation.get('subset_level')}"
            )
    evaluation_by_ordinal = {
        ordinal: row for ordinal, row in enumerate(raw_evaluations)
    }

    enriched_evaluations: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_evaluations):
        row = dict(raw)
        level = int(row.get("subset_level", 0) or 0)
        point_hash = row.get("latent_hash")
        state = state_by_hash.get(str(point_hash)) if point_hash is not None else None
        transition = transition_by_evaluation.get(ordinal)
        parent_state = None
        parent_evaluation = None
        if transition is not None:
            parent_state = state_by_id[str(transition["parent_state_id"])]
            parent_ordinal = parent_state.get("source_evaluation_ordinal")
            if parent_ordinal is not None:
                parent_evaluation = evaluation_by_ordinal.get(int(parent_ordinal))

        row.update(
            {
                "evaluation_ordinal": ordinal,
                "evaluation_id": f"e{ordinal + 1:09d}",
                "paper_level": level + 1,
                "evaluation_role": (
                    "initial_monte_carlo" if transition is None else "conditional_proposal"
                ),
                "state_id": state.get("state_id") if state is not None else None,
                "transition_id": (
                    transition.get("transition_id") if transition is not None else None
                ),
                "parent_state_id": (
                    transition.get("parent_state_id") if transition is not None else None
                ),
                "root_state_id": state.get("root_state_id") if state is not None else None,
                "chain_seed_state_id": (
                    transition.get("chain_seed_state_id")
                    if transition is not None
                    else None
                ),
                "chain_id": transition.get("chain_id") if transition is not None else None,
                "wave_index": (
                    transition.get("wave_index") if transition is not None else None
                ),
                "accepted": (
                    bool(transition["accepted"]) if transition is not None else True
                ),
                "inside_conditional_event": (
                    transition.get("inside_conditional_event")
                    if transition is not None
                    else None
                ),
                "rejection_reason": (
                    transition.get("rejection_reason") if transition is not None else None
                ),
                "conditioning_limit_state_threshold": (
                    transition.get("conditioning_limit_state_threshold")
                    if transition is not None
                    else None
                ),
                "right_censored": bool(row.get("hit_cap", False)),
                "termination_class": (
                    "generation_cap" if bool(row.get("hit_cap", False)) else "stop_or_eos"
                ),
                "censoring_lower_bound_tokens": (
                    int(row["output_tokens"]) if bool(row.get("hit_cap", False)) else None
                ),
                "prompt_character_count": len(str(row.get("input_text", ""))),
                "surrogate_prompt_token_count": len(row.get("input_token_ids", [])),
            }
        )
        if transition is not None:
            selected_threshold = float(getattr(obj, "performance_threshold", obj.threshold))
            row["conditioning_performance_threshold"] = (
                selected_threshold
                - float(transition["conditioning_limit_state_threshold"])
            )
            row["performance_overshoot_above_conditioning_threshold"] = (
                float(row["performance_score"])
                - float(row["conditioning_performance_threshold"])
            )
        else:
            row["conditioning_performance_threshold"] = None
            row["performance_overshoot_above_conditioning_threshold"] = None

        if parent_evaluation is not None:
            parent_tokens = int(parent_evaluation["output_tokens"])
            parent_repetition = float(parent_evaluation["repetition_score"])
            mismatch_count, mismatch_fraction = _prompt_token_distance(
                list(parent_evaluation.get("input_token_ids", [])),
                list(row.get("input_token_ids", [])),
            )
            row.update(
                {
                    "parent_evaluation_ordinal": int(
                        parent_state["source_evaluation_ordinal"]
                    ),
                    "parent_output_tokens": parent_tokens,
                    "output_token_change_from_parent": (
                        int(row["output_tokens"]) - parent_tokens
                    ),
                    "parent_repetition_score": parent_repetition,
                    "repetition_score_change_from_parent": (
                        float(row["repetition_score"]) - parent_repetition
                    ),
                    "parent_input_hash": parent_evaluation.get("input_hash"),
                    "parent_output_hash": parent_evaluation.get("output_hash"),
                    "prompt_token_change_count": mismatch_count,
                    "prompt_token_change_fraction": mismatch_fraction,
                    "same_prompt_as_parent": (
                        row.get("input_hash") == parent_evaluation.get("input_hash")
                    ),
                    "same_output_as_parent": (
                        row.get("output_hash") == parent_evaluation.get("output_hash")
                    ),
                }
            )
        else:
            row.update(
                {
                    "parent_evaluation_ordinal": None,
                    "parent_output_tokens": None,
                    "output_token_change_from_parent": None,
                    "parent_repetition_score": None,
                    "repetition_score_change_from_parent": None,
                    "parent_input_hash": None,
                    "parent_output_hash": None,
                    "prompt_token_change_count": None,
                    "prompt_token_change_fraction": None,
                    "same_prompt_as_parent": None,
                    "same_output_as_parent": None,
                }
            )
        enriched_evaluations.append(row)

    enriched_by_ordinal = {
        int(row["evaluation_ordinal"]): row for row in enriched_evaluations
    }
    for transition in transitions:
        evaluation_ordinal = transition.get("evaluation_ordinal")
        proposal = (
            enriched_by_ordinal.get(int(evaluation_ordinal))
            if evaluation_ordinal is not None
            else None
        )
        parent_state = state_by_id[str(transition["parent_state_id"])]
        parent_ordinal = parent_state.get("source_evaluation_ordinal")
        parent = (
            enriched_by_ordinal.get(int(parent_ordinal))
            if parent_ordinal is not None
            else None
        )
        selected_threshold = float(getattr(obj, "performance_threshold", obj.threshold))
        transition["conditioning_performance_threshold"] = (
            selected_threshold
            - float(transition["conditioning_limit_state_threshold"])
        )
        transition["parent_output_tokens"] = (
            int(parent["output_tokens"]) if parent is not None else None
        )
        transition["proposal_output_tokens"] = (
            int(proposal["output_tokens"]) if proposal is not None else None
        )
        transition["output_token_change_from_parent"] = (
            int(proposal["output_tokens"]) - int(parent["output_tokens"])
            if proposal is not None and parent is not None
            else None
        )
        transition["parent_repetition_score"] = (
            float(parent["repetition_score"]) if parent is not None else None
        )
        transition["proposal_repetition_score"] = (
            float(proposal["repetition_score"]) if proposal is not None else None
        )
        transition["repetition_score_change_from_parent"] = (
            float(proposal["repetition_score"]) - float(parent["repetition_score"])
            if proposal is not None and parent is not None
            else None
        )
        transition["proposal_performance_score"] = (
            float(proposal["performance_score"]) if proposal is not None else None
        )
        transition["performance_overshoot_above_conditioning_threshold"] = (
            float(proposal["performance_score"])
            - float(transition["conditioning_performance_threshold"])
            if proposal is not None
            else None
        )

    for state in states:
        source_ordinal = state.get("source_evaluation_ordinal")
        source = (
            enriched_by_ordinal.get(int(source_ordinal))
            if source_ordinal is not None
            else None
        )
        if source is not None:
            state.update(
                {
                    "source_evaluation_id": source["evaluation_id"],
                    "output_tokens": source["output_tokens"],
                    "repetition_score": source["repetition_score"],
                    "rep_2": source["rep_2"],
                    "rep_3": source["rep_3"],
                    "rep_4": source["rep_4"],
                    "performance_metric": source["performance_metric"],
                    "performance_score": source["performance_score"],
                    "target_threshold_exceeded": source["success"],
                    "hit_generation_cap": source["hit_cap"],
                    "input_hash": source["input_hash"],
                    "output_hash": source["output_hash"],
                }
            )

    state_by_id = {str(row["state_id"]): row for row in states}
    for slot in population_slots:
        state = state_by_id[str(slot["state_id"])]
        for key in (
            "source_evaluation_ordinal",
            "output_tokens",
            "repetition_score",
            "rep_2",
            "rep_3",
            "rep_4",
            "performance_metric",
            "performance_score",
            "target_threshold_exceeded",
            "hit_generation_cap",
            "input_hash",
            "output_hash",
            "parent_state_id",
            "root_state_id",
            "chain_seed_state_id",
            "chain_id",
            "wave_index",
        ):
            slot[key] = state.get(key)

    return enriched_evaluations, transitions, states, population_slots


def _level_payload(
    *,
    result: Any,
    obj: ThinkTrapObjective,
    evaluations: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    population_slots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations_by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    transitions_by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    slots_by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluations:
        evaluations_by_level[int(row["subset_level"] or 0)].append(row)
    for row in transitions:
        transitions_by_level[int(row["subset_level"])].append(row)
    for row in population_slots:
        slots_by_level[int(row["subset_level"])].append(row)

    payload: list[dict[str, Any]] = []
    selected_threshold = float(getattr(obj, "performance_threshold", obj.threshold))
    for summary in result.level_summaries:
        level = int(summary.level)
        level_evaluations = evaluations_by_level[level]
        level_transitions = transitions_by_level[level]
        slots = slots_by_level[level]
        state_rows: dict[str, dict[str, Any]] = {}
        for slot in slots:
            state_rows.setdefault(str(slot["state_id"]), slot)
        new_states = [
            row
            for row in state_rows.values()
            if int(row["first_seen_subset_level"]) == level
        ]
        accepted = [row for row in level_transitions if bool(row["accepted"])]
        accepted_with_parent = [
            row for row in accepted if row.get("output_token_change_from_parent") is not None
        ]
        longer_count = sum(
            int(row["output_token_change_from_parent"] > 0)
            for row in accepted_with_parent
        )
        equal_count = sum(
            int(row["output_token_change_from_parent"] == 0)
            for row in accepted_with_parent
        )
        shorter_count = sum(
            int(row["output_token_change_from_parent"] < 0)
            for row in accepted_with_parent
        )
        proposal_count = len(level_transitions)
        evaluated_count = sum(bool(row["model_evaluated"]) for row in level_transitions)
        item = {
            "subset_level": level,
            "paper_level": level + 1,
            "conditioning_performance_threshold": (
                selected_threshold - float(result.level_summaries[level - 1].threshold)
                if level > 0
                else None
            ),
            "selected_next_performance_threshold": (
                selected_threshold - float(summary.threshold)
            ),
            "population_size": int(summary.population_size),
            "retained_state_count": len(state_rows),
            "new_state_count": len(new_states),
            "retained_slot_output_tokens": _distribution(
                row.get("output_tokens") for row in slots
            ),
            "retained_state_output_tokens": _distribution(
                row.get("output_tokens") for row in state_rows.values()
            ),
            "new_state_output_tokens": _distribution(
                row.get("output_tokens") for row in new_states
            ),
            "retained_slot_repetition_score": _distribution(
                row.get("repetition_score") for row in slots
            ),
            "retained_state_repetition_score": _distribution(
                row.get("repetition_score") for row in state_rows.values()
            ),
            "new_state_repetition_score": _distribution(
                row.get("repetition_score") for row in new_states
            ),
            "evaluated_proposal_output_tokens": _distribution(
                row.get("output_tokens") for row in level_evaluations
            ),
            "proposal_count": proposal_count,
            "model_evaluated_proposal_count": evaluated_count,
            "accepted_proposal_count": len(accepted),
            "indicator_acceptance_rate": (
                len(accepted) / proposal_count if proposal_count else None
            ),
            "prior_kernel_no_change_count": sum(
                not bool(row["proposal_changed"]) for row in level_transitions
            ),
            "conditional_event_rejection_count": sum(
                row.get("rejection_reason") == "outside_conditional_event"
                for row in level_transitions
            ),
            "accepted_output_token_change_from_parent": _distribution(
                row.get("output_token_change_from_parent") for row in accepted
            ),
            "accepted_repetition_change_from_parent": _distribution(
                row.get("repetition_score_change_from_parent") for row in accepted
            ),
            "accepted_longer_than_parent_count": longer_count,
            "accepted_equal_to_parent_count": equal_count,
            "accepted_shorter_than_parent_count": shorter_count,
            "accepted_longer_than_parent_fraction": (
                longer_count / len(accepted_with_parent) if accepted_with_parent else None
            ),
            "accepted_threshold_overshoot": _distribution(
                row.get("performance_overshoot_above_conditioning_threshold")
                for row in accepted
            ),
            "active_accepting_chain_count": len(
                {int(row["chain_id"]) for row in accepted}
            ),
            "retained_target_hit_slot_count": sum(
                bool(row.get("target_threshold_exceeded")) for row in slots
            ),
            "retained_target_hit_state_count": sum(
                bool(row.get("target_threshold_exceeded"))
                for row in state_rows.values()
            ),
            "retained_cap_hit_slot_count": sum(
                bool(row.get("hit_generation_cap")) for row in slots
            ),
            "retained_cap_hit_state_count": sum(
                bool(row.get("hit_generation_cap")) for row in state_rows.values()
            ),
        }
        payload.append(item)
    return payload


def save_subset_trace(
    *,
    path: str,
    observer: SubsetTraceObserver | None,
    obj: ThinkTrapObjective,
    args: argparse.Namespace,
    result: Any,
) -> None:
    if not path or observer is None:
        return
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations, transitions, states, population_slots = _enrich_trace(observer, obj)
    levels = _level_payload(
        result=result,
        obj=obj,
        evaluations=evaluations,
        transitions=transitions,
        population_slots=population_slots,
    )

    state_ids = np.asarray([row["state_id"] for row in states], dtype="U16")
    vector_path = output_dir / "latent_states.npz"
    vector_temporary = output_dir / "latent_states.tmp.npz"
    np.savez_compressed(
        vector_temporary,
        state_ids=state_ids,
        latent_vectors=observer.state_vectors,
    )
    vector_temporary.replace(vector_path)

    files = {
        "evaluations": "evaluations.jsonl",
        "transitions": "transitions.jsonl",
        "states": "states.jsonl",
        "population_slots": "population_slots.jsonl",
        "latent_states": "latent_states.npz",
        "levels": "levels.json",
    }
    manifest = {
        "artifact_schema_version": TRACE_ARTIFACT_SCHEMA_VERSION,
        "observer": "read_only_subset_trace_observer_v1",
        "compute_path_invariance": (
            "Observer callbacks receive copies after proposal and acceptance decisions; "
            "they do not modify RNG state, model calls, thresholds, acceptance, or adaptation."
        ),
        "level_indexing": {
            "subset_level": "zero_based",
            "paper_level": "one_based",
        },
        "tested_llm_model": args.tested_llm_model,
        "surrogate_token_model": args.surrogate_token_model,
        "tested_model_revision_requested": getattr(args, "tested_model_revision", None),
        "surrogate_model_revision_requested": getattr(
            args, "surrogate_model_revision", None
        ),
        "estimator_seed": int(args.seed),
        "projection_seed": PROJECTION_SEED,
        "projection_mode": getattr(args, "projection_mode", None),
        "input_prompt_len": int(args.input_prompt_len),
        "ss_sample_space_dim": int(args.ss_sample_space_dim),
        "ss_samples_per_level": int(args.ss_samples_per_level),
        "ss_conditional_prob": float(args.ss_conditional_prob),
        "ss_max_levels": int(args.ss_max_levels),
        "performance_metric": str(getattr(obj, "performance_metric", "output_length")),
        "performance_threshold": float(getattr(obj, "performance_threshold", obj.threshold)),
        "llm_output_len_threshold": int(args.llm_output_len_threshold),
        "repetition_score_threshold": float(
            getattr(args, "repetition_score_threshold", obj.repetition_score_threshold)
        ),
        "llm_output_sampler": str(args.llm_output_sampler),
        "chat_template_thinking": str(getattr(args, "chat_template_thinking", "auto")),
        "generation_batch_size": int(getattr(args, "generation_batch_size", 1)),
        "objective_repeats": int(getattr(args, "objective_repeats", 1)),
        "objective_aggregation": str(getattr(args, "objective_aggregation", "mean")),
        "max_new_tokens": int(getattr(obj.generator, "max_new_tokens", 0)),
        "stop_reason": result.stop_reason.value,
        "estimated_probability": float(result.estimated_probability),
        "target_reached": bool(result.target_reached),
        "reliable_probability": bool(result.reliable_probability),
        "record_counts": {
            "evaluations": len(evaluations),
            "transitions": len(transitions),
            "states": len(states),
            "population_slots": len(population_slots),
            "levels": len(levels),
        },
        "files": files,
        "runtime": runtime_metadata(getattr(obj.generator, "device", None)),
    }
    manifest.update(dict(getattr(obj, "run_metadata", {})))
    manifest["projection_seed"] = PROJECTION_SEED

    _write_jsonl(output_dir / files["evaluations"], evaluations)
    _write_jsonl(output_dir / files["transitions"], transitions)
    _write_jsonl(output_dir / files["states"], states)
    _write_jsonl(output_dir / files["population_slots"], population_slots)
    _write_json(output_dir / files["levels"], {"levels": levels})
    manifest["file_sha256"] = {
        name: _sha256(output_dir / filename) for name, filename in files.items()
    }
    _write_json(output_dir / "manifest.json", manifest)
    checkpoint_path = output_dir / "checkpoint" / "checkpoint.json"
    if checkpoint_path.parent.exists():
        _write_json(
            checkpoint_path,
            {
                "artifact_schema_version": TRACE_ARTIFACT_SCHEMA_VERSION,
                "complete": True,
                "completed_level_count": len(levels),
                "highest_completed_subset_level": (
                    int(levels[-1]["subset_level"]) if levels else None
                ),
                "final_manifest": "../manifest.json",
            },
        )
    print(
        "[subset-trace] "
        f"evaluations={len(evaluations)} transitions={len(transitions)} "
        f"states={len(states)} population_slots={len(population_slots)} "
        f"path={output_dir}",
        flush=True,
    )
