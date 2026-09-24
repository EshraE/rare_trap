"""Read-only subset-sampler trace collection.

The observer receives copies of sampler state after decisions have been made. It
does not participate in proposal generation, target evaluation, acceptance, or
adaptation.
"""

from __future__ import annotations

import math
from collections import Counter
from hashlib import blake2b
from typing import Any

import numpy as np


def latent_hash(point: np.ndarray) -> str:
    """Match the objective's stable, rounded latent-point identifier."""
    array = np.asarray(point, dtype=np.float32)
    rounded = np.round(array, decimals=6).astype(np.float32, copy=False)
    return blake2b(rounded.tobytes(), digest_size=16).hexdigest()


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(np.dot(left, right) / (left_norm * right_norm))


class SubsetTraceObserver:
    """Collect exact state lineage, transitions, and retained populations."""

    schema_version = 1

    def __init__(self) -> None:
        self.states: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self.population_slots: list[dict[str, Any]] = []
        self.population_summaries: list[dict[str, Any]] = []
        self._state_by_hash: dict[str, str] = {}
        self._state_record_by_id: dict[str, dict[str, Any]] = {}
        self._state_vectors: list[np.ndarray] = []
        self._chain_seed_by_level_chain: dict[tuple[int, int], str] = {}
        self._next_state_number = 1
        self._next_transition_number = 1
        self._evaluation_ordinal = 0

    @property
    def state_vectors(self) -> np.ndarray:
        if not self._state_vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(self._state_vectors).astype(np.float32, copy=False)

    @property
    def observed_evaluation_count(self) -> int:
        return int(self._evaluation_ordinal)

    def _new_state_id(self) -> str:
        state_id = f"s{self._next_state_number:09d}"
        self._next_state_number += 1
        return state_id

    def _register_state(
        self,
        *,
        point: np.ndarray,
        level: int,
        origin: str,
        limit_state: float,
        log_target: float,
        parent_state_id: str | None = None,
        chain_seed_state_id: str | None = None,
        chain_id: int | None = None,
        wave_index: int | None = None,
        evaluation_ordinal: int | None = None,
    ) -> str:
        point_hash = latent_hash(point)
        existing = self._state_by_hash.get(point_hash)
        if existing is not None:
            return existing

        state_id = self._new_state_id()
        if parent_state_id is None:
            root_state_id = state_id
        else:
            parent = self._state_record_by_id[parent_state_id]
            root_state_id = str(parent["root_state_id"])
        vector_index = len(self._state_vectors)
        vector = np.asarray(point, dtype=np.float32).copy()
        self._state_vectors.append(vector)
        record = {
            "state_id": state_id,
            "latent_hash": point_hash,
            "latent_vector_index": vector_index,
            "latent_norm": float(np.linalg.norm(vector)),
            "first_seen_subset_level": int(level),
            "first_seen_paper_level": int(level) + 1,
            "origin": str(origin),
            "parent_state_id": parent_state_id,
            "root_state_id": root_state_id,
            "chain_seed_state_id": chain_seed_state_id,
            "chain_id": chain_id,
            "wave_index": wave_index,
            "source_evaluation_ordinal": evaluation_ordinal,
            "limit_state": float(limit_state),
            "log_target": float(log_target),
        }
        self.states.append(record)
        self._state_by_hash[point_hash] = state_id
        self._state_record_by_id[state_id] = record
        return state_id

    def observe_population(
        self,
        *,
        level: int,
        points: np.ndarray,
        values: np.ndarray,
        log_targets: np.ndarray,
        elite_count: int,
        threshold: float,
    ) -> None:
        points = np.asarray(points, dtype=float)
        values = np.asarray(values, dtype=float)
        log_targets = np.asarray(log_targets, dtype=float)
        if int(level) == 0 and self._evaluation_ordinal == 0:
            self._evaluation_ordinal = int(points.shape[0])

        state_ids: list[str] = []
        for point, value, log_target in zip(points, values, log_targets):
            point_hash = latent_hash(point)
            state_id = self._state_by_hash.get(point_hash)
            if state_id is None:
                state_id = self._register_state(
                    point=point,
                    level=int(level),
                    origin="initial_monte_carlo" if int(level) == 0 else "recovered",
                    limit_state=float(value),
                    log_target=float(log_target),
                )
            state_ids.append(state_id)

        multiplicities = Counter(state_ids)
        for population_index, (state_id, value, log_target) in enumerate(
            zip(state_ids, values, log_targets)
        ):
            state = self._state_record_by_id[state_id]
            self.population_slots.append(
                {
                    "subset_level": int(level),
                    "paper_level": int(level) + 1,
                    "population_index": int(population_index),
                    "performance_rank": int(population_index) + 1,
                    "state_id": state_id,
                    "latent_hash": state["latent_hash"],
                    "state_multiplicity": int(multiplicities[state_id]),
                    "first_seen_subset_level": state["first_seen_subset_level"],
                    "carried_from_earlier_level": (
                        int(state["first_seen_subset_level"]) < int(level)
                    ),
                    "selected_as_next_seed": int(population_index) < int(elite_count),
                    "limit_state": float(value),
                    "log_target": float(log_target),
                }
            )
        self.population_summaries.append(
            {
                "subset_level": int(level),
                "paper_level": int(level) + 1,
                "population_size": len(state_ids),
                "distinct_state_count": len(multiplicities),
                "duplicate_slot_fraction": (
                    1.0 - len(multiplicities) / max(len(state_ids), 1)
                ),
                "elite_count": int(elite_count),
                "limit_state_threshold": float(threshold),
            }
        )

    def observe_transition_batch(
        self,
        *,
        level: int,
        wave_index: int,
        threshold: float,
        proposal_half_width: float,
        parent_points: np.ndarray,
        parent_values: np.ndarray,
        parent_log_targets: np.ndarray,
        proposal_points: np.ndarray,
        proposal_values: np.ndarray,
        proposal_log_targets: np.ndarray,
        changed_mask: np.ndarray,
        accepted_mask: np.ndarray,
    ) -> None:
        arrays = (
            np.asarray(parent_points, dtype=float),
            np.asarray(parent_values, dtype=float),
            np.asarray(parent_log_targets, dtype=float),
            np.asarray(proposal_points, dtype=float),
            np.asarray(proposal_values, dtype=float),
            np.asarray(proposal_log_targets, dtype=float),
            np.asarray(changed_mask, dtype=bool),
            np.asarray(accepted_mask, dtype=bool),
        )
        (
            parent_points,
            parent_values,
            parent_log_targets,
            proposal_points,
            proposal_values,
            proposal_log_targets,
            changed_mask,
            accepted_mask,
        ) = arrays

        for chain_id in range(int(parent_points.shape[0])):
            parent_point = parent_points[chain_id]
            proposal_point = proposal_points[chain_id]
            parent_hash = latent_hash(parent_point)
            parent_state_id = self._state_by_hash[parent_hash]
            chain_key = (int(level), int(chain_id))
            chain_seed_state_id = self._chain_seed_by_level_chain.setdefault(
                chain_key, parent_state_id
            )
            changed = bool(changed_mask[chain_id])
            accepted = bool(accepted_mask[chain_id])
            evaluation_ordinal = None
            if changed:
                evaluation_ordinal = self._evaluation_ordinal
                self._evaluation_ordinal += 1

            proposal_state_id = None
            if accepted:
                proposal_state_id = self._register_state(
                    point=proposal_point,
                    level=int(level),
                    origin="accepted_proposal",
                    limit_state=float(proposal_values[chain_id]),
                    log_target=float(proposal_log_targets[chain_id]),
                    parent_state_id=parent_state_id,
                    chain_seed_state_id=chain_seed_state_id,
                    chain_id=int(chain_id),
                    wave_index=int(wave_index),
                    evaluation_ordinal=evaluation_ordinal,
                )
            result_state_id = proposal_state_id if accepted else parent_state_id
            delta = proposal_point - parent_point
            changed_coordinates = int(np.count_nonzero(delta))
            if accepted:
                rejection_reason = None
            elif not changed:
                rejection_reason = "prior_kernel_rejected"
            else:
                rejection_reason = "outside_conditional_event"

            self.transitions.append(
                {
                    "transition_id": f"t{self._next_transition_number:09d}",
                    "subset_level": int(level),
                    "paper_level": int(level) + 1,
                    "wave_index": int(wave_index),
                    "chain_id": int(chain_id),
                    "chain_seed_state_id": chain_seed_state_id,
                    "parent_state_id": parent_state_id,
                    "parent_latent_hash": parent_hash,
                    "proposal_latent_hash": latent_hash(proposal_point),
                    "proposal_state_id": proposal_state_id,
                    "result_state_id": result_state_id,
                    "evaluation_ordinal": evaluation_ordinal,
                    "proposal_changed": changed,
                    "model_evaluated": changed,
                    "inside_conditional_event": (
                        changed and float(proposal_values[chain_id]) <= float(threshold)
                    ),
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                    "conditioning_limit_state_threshold": float(threshold),
                    "proposal_half_width": float(proposal_half_width),
                    "parent_limit_state": float(parent_values[chain_id]),
                    "proposal_limit_state": float(proposal_values[chain_id]),
                    "limit_state_change": float(
                        proposal_values[chain_id] - parent_values[chain_id]
                    ),
                    "parent_log_target": float(parent_log_targets[chain_id]),
                    "proposal_log_target": float(proposal_log_targets[chain_id]),
                    "changed_coordinate_count": changed_coordinates,
                    "latent_l2_step": float(np.linalg.norm(delta)),
                    "latent_linf_step": float(np.max(np.abs(delta))) if delta.size else 0.0,
                    "parent_proposal_cosine_similarity": _cosine_similarity(
                        parent_point, proposal_point
                    ),
                }
            )
            self._next_transition_number += 1
