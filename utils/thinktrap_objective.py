"""ThinkTrap objective mapping latent points to selected performance limit states."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch

from utils.llm_generator import HFTextGenerator
from utils.projection_space import ProjectedPrompt, ProjectionSpace
from utils.repetition_metrics import repetition_metrics


@dataclass
class GenerationMetrics:
    length: int
    normalized_length: float
    performance_metric: str
    performance_score: float
    performance_threshold: float
    performance_margin: float
    repetition_score: float
    rep_2: float
    rep_3: float
    rep_4: float
    elapsed_sec: float
    tokens_per_second: float
    success: bool
    hit_cap: bool


@dataclass
class EvaluationRecord:
    sample_index: int
    subset_level: Optional[int]
    length: int
    output_token_margin: int
    success: bool
    hit_cap: bool
    elapsed_sec: float
    tokens_per_second: float
    input_token_count: int
    rendered_input_text: str
    completion_text: Optional[str]
    input_token_ids: list[int] = field(default_factory=list)
    repeat_lengths: list[int] = field(default_factory=list)
    repeat_aggregation: str = "single"
    repeat_count: int = 1
    best_repeat_index: Optional[int] = None
    performance_metric: str = "output_length"
    performance_score: Optional[float] = None
    performance_threshold: Optional[float] = None
    performance_margin: Optional[float] = None
    repetition_score: Optional[float] = None
    rep_2: Optional[float] = None
    rep_3: Optional[float] = None
    rep_4: Optional[float] = None


@dataclass
class ThinkTrapObjective:
    generator: HFTextGenerator
    projection_space: ProjectionSpace
    threshold: int
    generation_batch_size: int = 1
    max_saved_records: int = 10
    max_saved_output_text_records: int = 2
    objective_repeats: int = 1
    objective_aggregation: str = "mean"
    performance_metric: str = "output_length"
    repetition_score_threshold: float = 0.9
    raw_samples_path: str = ""
    run_metadata: dict[str, Any] = field(default_factory=dict)
    sample_counter: int = field(default=0, init=False)
    progress_line_width: int = field(default=0, init=False)
    progress_token_step: int = field(default=500, init=False)
    progress_last_tokens: dict[int, int] = field(default_factory=dict, init=False)
    best_records: List[EvaluationRecord] = field(default_factory=list, init=False)
    evaluation_trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    record_count: int = field(default=0, init=False)
    unique_input_hashes: set[bytes] = field(default_factory=set, init=False)
    unique_output_hashes: set[bytes] = field(default_factory=set, init=False)
    unique_success_input_hashes: set[bytes] = field(default_factory=set, init=False)
    unique_success_output_hashes: set[bytes] = field(default_factory=set, init=False)
    unique_z_hashes: set[bytes] = field(default_factory=set, init=False)
    evaluation_metrics_by_latent_key: dict[bytes, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
    )
    level_eval_counts: dict[int, int] = field(default_factory=dict, init=False)
    level_input_hashes: dict[int, set[bytes]] = field(default_factory=dict, init=False)
    level_output_hashes: dict[int, set[bytes]] = field(default_factory=dict, init=False)
    level_success_input_hashes: dict[int, set[bytes]] = field(default_factory=dict, init=False)
    level_success_output_hashes: dict[int, set[bytes]] = field(default_factory=dict, init=False)
    level_output_hash_counts: dict[int, Counter[bytes]] = field(default_factory=dict, init=False)
    current_subset_level: Optional[int] = field(default=None, init=False)

    RAW_SAMPLE_SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        if int(self.threshold) < 1:
            raise ValueError("threshold must be positive.")
        if int(self.generation_batch_size) < 1:
            raise ValueError("generation_batch_size must be at least 1.")
        if int(self.objective_repeats) < 1:
            raise ValueError("objective_repeats must be at least 1.")
        if str(self.objective_aggregation) not in {"mean", "median", "max", "p90", "top20mean"}:
            raise ValueError(f"Unknown objective_aggregation={self.objective_aggregation!r}.")
        if str(self.performance_metric) not in {"output_length", "repetition_score"}:
            raise ValueError(f"Unknown performance_metric={self.performance_metric!r}.")
        if not (0.0 <= float(self.repetition_score_threshold) <= 1.0):
            raise ValueError("repetition_score_threshold must be in [0, 1].")

    @property
    def Dg(self) -> int:
        return int(self.projection_space.d_latent)

    @staticmethod
    def _text_hash(text: str) -> bytes:
        return blake2b(text.encode("utf-8", errors="replace"), digest_size=16).digest()

    @staticmethod
    def _optional_text_hash(text: Optional[str]) -> bytes:
        return ThinkTrapObjective._text_hash("" if text is None else text)

    @staticmethod
    def _latent_hash(latent_point: Optional[np.ndarray]) -> Optional[bytes]:
        if latent_point is None:
            return None
        array = np.asarray(latent_point, dtype=np.float32)
        if int(array.size) == 0:
            return None
        rounded = np.round(array, decimals=6).astype(np.float32, copy=False)
        return blake2b(rounded.tobytes(), digest_size=16).digest()

    @staticmethod
    def _latent_metric_key(latent_point: Optional[np.ndarray]) -> Optional[bytes]:
        if latent_point is None:
            return None
        array = np.asarray(latent_point, dtype=np.float32)
        if int(array.size) == 0:
            return None
        return blake2b(array.tobytes(), digest_size=16).digest()

    def _store_latent_metrics(
        self,
        *,
        latent_point: Optional[np.ndarray],
        sample_index: int,
        metrics: GenerationMetrics,
    ) -> None:
        key = self._latent_metric_key(latent_point)
        if key is None:
            return
        self.evaluation_metrics_by_latent_key[key] = {
            "source_sample_index": int(sample_index),
            "source_subset_level": self.current_subset_level,
            "output_tokens": int(metrics.length),
            "repetition_score": float(metrics.repetition_score),
            "rep_2": float(metrics.rep_2),
            "rep_3": float(metrics.rep_3),
            "rep_4": float(metrics.rep_4),
            "selected_performance_metric": str(metrics.performance_metric),
            "selected_performance_score": float(metrics.performance_score),
            "selected_performance_threshold": float(metrics.performance_threshold),
            "selected_threshold_exceeded": bool(metrics.success),
            "hit_generation_cap": bool(metrics.hit_cap),
        }

    def metrics_for_latent_point(self, latent_point: np.ndarray) -> Optional[dict[str, Any]]:
        key = self._latent_metric_key(latent_point)
        if key is None:
            return None
        metrics = self.evaluation_metrics_by_latent_key.get(key)
        return None if metrics is None else dict(metrics)

    @staticmethod
    def _record_sort_key(record: EvaluationRecord) -> tuple[int, int, int]:
        score = (
            float(record.performance_score)
            if record.performance_score is not None
            else float(record.length)
        )
        return (-int(record.success), -int(round(1_000_000.0 * score)), int(record.sample_index))

    @property
    def performance_threshold(self) -> float:
        if self.performance_metric == "output_length":
            return float(self.threshold)
        if self.performance_metric == "repetition_score":
            return float(self.repetition_score_threshold)
        raise ValueError(f"Unknown performance_metric={self.performance_metric!r}.")

    def _special_token_ids(self) -> set[int]:
        return {
            int(token_id)
            for token_id in getattr(self.generator.tokenizer, "all_special_ids", [])
            if token_id is not None
        }

    def _prune_saved_output_texts(self) -> None:
        keep_count = max(0, int(self.max_saved_output_text_records))
        for index, record in enumerate(self.best_records):
            if index >= keep_count:
                record.completion_text = None

    def _store_record(self, record: EvaluationRecord) -> None:
        capacity = max(1, int(self.max_saved_records))
        existing_index = next(
            (
                index
                for index, existing in enumerate(self.best_records)
                if existing.rendered_input_text == record.rendered_input_text
            ),
            None,
        )
        if existing_index is not None:
            if self._record_sort_key(record) >= self._record_sort_key(self.best_records[existing_index]):
                return
            self.best_records[existing_index] = record
        elif len(self.best_records) < capacity:
            self.best_records.append(record)
        elif self._record_sort_key(record) < self._record_sort_key(self.best_records[-1]):
            self.best_records[-1] = record
        else:
            return
        self.best_records.sort(key=self._record_sort_key)
        self._prune_saved_output_texts()

    def _store_evaluation_trace(
        self,
        record: EvaluationRecord,
        *,
        input_hash: bytes,
        output_hash: bytes,
        latent_hash: Optional[bytes],
    ) -> None:
        self.evaluation_trace.append(
            {
                "sample_index": int(record.sample_index),
                "subset_level": record.subset_level,
                "output_tokens": int(record.length),
                "output_token_margin": int(record.output_token_margin),
                "performance_metric": str(record.performance_metric),
                "performance_score": record.performance_score,
                "performance_threshold": record.performance_threshold,
                "performance_margin": record.performance_margin,
                "repetition_score": record.repetition_score,
                "rep_2": record.rep_2,
                "rep_3": record.rep_3,
                "rep_4": record.rep_4,
                "success": bool(record.success),
                "hit_cap": bool(record.hit_cap),
                "elapsed_sec": float(record.elapsed_sec),
                "tokens_per_second": float(record.tokens_per_second),
                "input_token_count": int(record.input_token_count),
                "input_token_ids": list(record.input_token_ids),
                "input_text": record.rendered_input_text,
                "input_hash": input_hash.hex(),
                "output_hash": output_hash.hex(),
                "latent_hash": latent_hash.hex() if latent_hash is not None else None,
                "repeat_count": int(record.repeat_count),
                "repeat_aggregation": str(record.repeat_aggregation),
                "repeat_lengths": list(record.repeat_lengths),
            }
        )

    def initialize_raw_sample_stream(self, metadata: dict[str, Any]) -> None:
        """Create a new lossless, append-only sample stream for this run."""
        if not str(self.raw_samples_path):
            return
        output_path = Path(self.raw_samples_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.stat().st_size > 0:
            raise FileExistsError(f"Refusing to overwrite raw sample stream: {output_path}")
        header = {
            "record_type": "metadata",
            "artifact_schema_version": self.RAW_SAMPLE_SCHEMA_VERSION,
            "description": (
                "Lossless append-only model-evaluation stream. Each sample row stores "
                "the exact float32 latent vector, projected prompt, model input token IDs, "
                "completion token IDs/text, and per-generation metrics."
            ),
            "special_token_ids": sorted(self._special_token_ids()),
            **metadata,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _append_raw_sample(
        self,
        *,
        record: EvaluationRecord,
        latent_point: Optional[np.ndarray],
        results: list[Any],
        result_metrics: list[GenerationMetrics],
        input_hash: bytes,
        latent_hash: Optional[bytes],
    ) -> None:
        if not str(self.raw_samples_path):
            return
        if len(results) != len(result_metrics):
            raise ValueError("Raw-sample results and metrics must have equal length.")
        generations = []
        for repeat_index, (result, metrics) in enumerate(zip(results, result_metrics)):
            output_text = str(result.completion_text)
            generations.append(
                {
                    "repeat_index": int(repeat_index),
                    "model_input_token_ids": [
                        int(token_id) for token_id in result.model_input_token_ids
                    ],
                    "input_token_count": int(result.input_token_count),
                    "completion_token_ids": [
                        int(token_id) for token_id in result.completion_token_ids
                    ],
                    "completion_text": output_text,
                    "output_hash": self._text_hash(output_text).hex(),
                    "output_tokens": int(result.completion_length),
                    "repetition_score": float(metrics.repetition_score),
                    "rep_2": float(metrics.rep_2),
                    "rep_3": float(metrics.rep_3),
                    "rep_4": float(metrics.rep_4),
                    "performance_score": float(metrics.performance_score),
                    "performance_threshold": float(metrics.performance_threshold),
                    "performance_margin": float(metrics.performance_margin),
                    "success": bool(metrics.success),
                    "hit_cap": bool(metrics.hit_cap),
                    "elapsed_sec": float(result.elapsed_sec),
                    "tokens_per_second": float(result.tokens_per_second),
                }
            )
        latent_values = (
            None
            if latent_point is None
            else np.asarray(latent_point, dtype=np.float32).reshape(-1).tolist()
        )
        row = {
            "record_type": "sample",
            "artifact_schema_version": self.RAW_SAMPLE_SCHEMA_VERSION,
            "sample_index": int(record.sample_index),
            "subset_level": record.subset_level,
            "latent_vector_float32": latent_values,
            "latent_hash": latent_hash.hex() if latent_hash is not None else None,
            "input_token_ids": [int(token_id) for token_id in record.input_token_ids],
            "input_text": record.rendered_input_text,
            "input_hash": input_hash.hex(),
            "repeat_count": int(record.repeat_count),
            "repeat_aggregation": str(record.repeat_aggregation),
            "best_repeat_index": record.best_repeat_index,
            "aggregate": {
                "output_tokens": int(record.length),
                "output_token_margin": int(record.output_token_margin),
                "performance_metric": str(record.performance_metric),
                "performance_score": record.performance_score,
                "performance_threshold": record.performance_threshold,
                "performance_margin": record.performance_margin,
                "repetition_score": record.repetition_score,
                "rep_2": record.rep_2,
                "rep_3": record.rep_3,
                "rep_4": record.rep_4,
                "success": bool(record.success),
                "hit_cap": bool(record.hit_cap),
                "elapsed_sec": float(record.elapsed_sec),
                "tokens_per_second": float(record.tokens_per_second),
            },
            "generations": generations,
        }
        output_path = Path(self.raw_samples_path)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def set_subset_level(self, level: int) -> None:
        self.current_subset_level = int(level)

    def _level_key(self) -> int:
        return -1 if self.current_subset_level is None else int(self.current_subset_level)

    @staticmethod
    def _level_set(mapping: dict[int, set[bytes]], level: int) -> set[bytes]:
        return mapping.setdefault(int(level), set())

    def level_diversity_stats(self, level: int) -> dict[str, Any]:
        level = int(level)
        eval_count = int(self.level_eval_counts.get(level, 0))
        input_count = len(self.level_input_hashes.get(level, set()))
        output_count = len(self.level_output_hashes.get(level, set()))
        success_input_count = len(self.level_success_input_hashes.get(level, set()))
        success_output_count = len(self.level_success_output_hashes.get(level, set()))
        output_counts = self.level_output_hash_counts.get(level, Counter())
        top_output_repeats = max(output_counts.values(), default=0)
        return {
            "level_eval_count": eval_count,
            "level_unique_input_count": input_count,
            "level_unique_output_count": output_count,
            "level_unique_success_input_count": success_input_count,
            "level_unique_success_output_count": success_output_count,
            "level_input_duplicate_rate": (
                1.0 - float(input_count) / float(eval_count) if eval_count else 0.0
            ),
            "level_output_duplicate_rate": (
                1.0 - float(output_count) / float(eval_count) if eval_count else 0.0
            ),
            "level_top_output_repeats": int(top_output_repeats),
        }

    def _write_progress_line(self, line: str, *, final: bool = False) -> None:
        if not sys.stdout.isatty():
            print(line, flush=True)
            return
        padding = " " * max(0, self.progress_line_width - len(line))
        print(f"\r{line}{padding}", end="\n" if final else "", flush=True)
        self.progress_line_width = max(self.progress_line_width, len(line))

    def _start_sample_progress(self) -> int:
        self.sample_counter += 1
        sample_index = int(self.sample_counter)
        self.progress_last_tokens[sample_index] = 0
        if sys.stdout.isatty():
            self._write_progress_line(f"[sample] progress count={sample_index} output_tokens=0")
        return sample_index

    def _reserve_sample_indices(self, count: int) -> list[int]:
        indices: list[int] = []
        for _ in range(int(count)):
            self.sample_counter += 1
            sample_index = int(self.sample_counter)
            self.progress_last_tokens[sample_index] = 0
            indices.append(sample_index)
        return indices

    def _update_sample_progress(self, sample_index: int, output_tokens: int, *, final: bool = False) -> None:
        sample_index = int(sample_index)
        output_tokens = int(output_tokens)
        last_tokens = int(self.progress_last_tokens.get(sample_index, 0))
        if not final and output_tokens - last_tokens < int(self.progress_token_step):
            return
        self.progress_last_tokens[sample_index] = output_tokens
        status = "done" if final else "progress"
        self._write_progress_line(
            f"[sample] {status} count={sample_index} output_tokens={output_tokens}",
            final=bool(final),
        )
        if final:
            self.progress_last_tokens.pop(sample_index, None)

    def _write_batch_summary(self, sample_indices: list[int], lengths: list[int]) -> None:
        if not sample_indices or not lengths:
            return
        length_array = np.asarray(lengths, dtype=np.int64)
        first_index = int(sample_indices[0])
        last_index = int(sample_indices[-1])
        length_threshold_hits = int(np.sum(length_array >= int(self.threshold)))
        self._write_progress_line(
            "[batch] done "
            f"samples={first_index}-{last_index} "
            f"size={len(lengths)} "
            f"min_output_tokens={int(length_array.min())} "
            f"max_output_tokens={int(length_array.max())} "
            f"mean_output_tokens={float(length_array.mean()):.1f} "
            f"length_threshold_hits={length_threshold_hits}",
            final=True,
        )
        for sample_index in sample_indices:
            self.progress_last_tokens.pop(int(sample_index), None)

    def _aggregate_repeat_lengths(self, lengths: list[int]) -> float:
        if not lengths:
            return 0.0
        values = np.asarray(lengths, dtype=np.float64)
        mode = str(self.objective_aggregation)
        if mode == "mean":
            return float(values.mean())
        if mode == "median":
            return float(np.median(values))
        if mode == "max":
            return float(values.max())
        if mode == "p90":
            return float(np.percentile(values, 90))
        if mode == "top20mean":
            keep = max(1, int(np.ceil(0.2 * values.size)))
            return float(np.sort(values)[-keep:].mean())
        raise ValueError(f"Unknown objective_aggregation={mode!r}.")

    def _aggregate_values(self, values: list[float]) -> float:
        if not values:
            return 0.0
        array = np.asarray(values, dtype=np.float64)
        mode = str(self.objective_aggregation)
        if mode == "mean":
            return float(array.mean())
        if mode == "median":
            return float(np.median(array))
        if mode == "max":
            return float(array.max())
        if mode == "p90":
            return float(np.percentile(array, 90))
        if mode == "top20mean":
            keep = max(1, int(np.ceil(0.2 * array.size)))
            return float(np.sort(array)[-keep:].mean())
        raise ValueError(f"Unknown objective_aggregation={mode!r}.")

    def _metrics_from_result(
        self,
        *,
        result,
        elapsed_sec: float,
        tokens_per_second: float,
        hit_cap: bool,
    ) -> GenerationMetrics:
        length = int(result.completion_length)
        rep = repetition_metrics(
            result.completion_token_ids,
            special_token_ids=self._special_token_ids(),
        )
        if self.performance_metric == "output_length":
            score = float(length)
        elif self.performance_metric == "repetition_score":
            score = float(rep.repetition_score)
        else:
            raise ValueError(f"Unknown performance_metric={self.performance_metric!r}.")
        threshold = self.performance_threshold
        return GenerationMetrics(
            length=length,
            normalized_length=float(length) / float(max(self.generator.max_new_tokens, 1)),
            performance_metric=str(self.performance_metric),
            performance_score=float(score),
            performance_threshold=float(threshold),
            performance_margin=float(score) - float(threshold),
            repetition_score=float(rep.repetition_score),
            rep_2=float(rep.rep_2),
            rep_3=float(rep.rep_3),
            rep_4=float(rep.rep_4),
            elapsed_sec=float(elapsed_sec),
            tokens_per_second=float(tokens_per_second),
            success=bool(float(score) >= float(threshold)),
            hit_cap=bool(hit_cap),
        )

    def _record_evaluation(
        self,
        *,
        sample_index: int,
        prompt: ProjectedPrompt,
        result,
        metrics: GenerationMetrics,
        latent_point: Optional[np.ndarray] = None,
    ) -> None:
        self.record_count += 1
        level = self._level_key()
        input_hash = self._text_hash(prompt.text)
        output_hash = self._optional_text_hash(result.completion_text)
        self.unique_input_hashes.add(input_hash)
        self.unique_output_hashes.add(output_hash)
        if bool(metrics.success):
            self.unique_success_input_hashes.add(input_hash)
            self.unique_success_output_hashes.add(output_hash)
        z_hash = self._latent_hash(latent_point)
        if z_hash is not None:
            self.unique_z_hashes.add(z_hash)
        self._store_latent_metrics(
            latent_point=latent_point,
            sample_index=sample_index,
            metrics=metrics,
        )
        self.level_eval_counts[level] = int(self.level_eval_counts.get(level, 0)) + 1
        self._level_set(self.level_input_hashes, level).add(input_hash)
        self._level_set(self.level_output_hashes, level).add(output_hash)
        output_counts = self.level_output_hash_counts.setdefault(level, Counter())
        output_counts[output_hash] += 1
        if bool(metrics.success):
            self._level_set(self.level_success_input_hashes, level).add(input_hash)
            self._level_set(self.level_success_output_hashes, level).add(output_hash)

        record = EvaluationRecord(
            sample_index=int(sample_index),
            subset_level=self.current_subset_level,
            length=int(result.completion_length),
            output_token_margin=int(result.completion_length) - int(self.threshold),
            success=bool(metrics.success),
            hit_cap=bool(metrics.hit_cap),
            elapsed_sec=float(metrics.elapsed_sec),
            tokens_per_second=float(metrics.tokens_per_second),
            input_token_count=int(result.input_token_count),
            input_token_ids=[int(token_id) for token_id in prompt.token_ids],
            rendered_input_text=prompt.text,
            completion_text=result.completion_text,
            performance_metric=str(metrics.performance_metric),
            performance_score=float(metrics.performance_score),
            performance_threshold=float(metrics.performance_threshold),
            performance_margin=float(metrics.performance_margin),
            repetition_score=float(metrics.repetition_score),
            rep_2=float(metrics.rep_2),
            rep_3=float(metrics.rep_3),
            rep_4=float(metrics.rep_4),
        )
        self._store_evaluation_trace(
            record,
            input_hash=input_hash,
            output_hash=output_hash,
            latent_hash=z_hash,
        )
        self._append_raw_sample(
            record=record,
            latent_point=latent_point,
            results=[result],
            result_metrics=[metrics],
            input_hash=input_hash,
            latent_hash=z_hash,
        )
        self._store_record(record)

    def _record_repeated_evaluation(
        self,
        *,
        sample_index: int,
        prompt: ProjectedPrompt,
        results: list,
        latent_point: Optional[np.ndarray] = None,
    ) -> GenerationMetrics:
        lengths = [int(result.completion_length) for result in results]
        aggregate_length = int(round(self._aggregate_repeat_lengths(lengths)))
        elapsed_sec = float(sum(float(result.elapsed_sec) for result in results))
        tokens_per_second = (
            float(sum(lengths)) / float(max(elapsed_sec, 1e-9)) if results else 0.0
        )
        per_result_metrics = [
            self._metrics_from_result(
                result=result,
                elapsed_sec=result.elapsed_sec,
                tokens_per_second=result.tokens_per_second,
                hit_cap=result.hit_cap,
            )
            for result in results
        ]
        aggregate_score = self._aggregate_values(
            [float(item.performance_score) for item in per_result_metrics]
        )
        metrics = GenerationMetrics(
            length=int(aggregate_length),
            normalized_length=float(aggregate_length) / float(max(self.generator.max_new_tokens, 1)),
            performance_metric=str(self.performance_metric),
            performance_score=float(aggregate_score),
            performance_threshold=float(self.performance_threshold),
            performance_margin=float(aggregate_score) - float(self.performance_threshold),
            repetition_score=self._aggregate_values(
                [float(item.repetition_score) for item in per_result_metrics]
            ),
            rep_2=self._aggregate_values([float(item.rep_2) for item in per_result_metrics]),
            rep_3=self._aggregate_values([float(item.rep_3) for item in per_result_metrics]),
            rep_4=self._aggregate_values([float(item.rep_4) for item in per_result_metrics]),
            elapsed_sec=elapsed_sec,
            tokens_per_second=tokens_per_second,
            success=bool(float(aggregate_score) >= float(self.performance_threshold)),
            hit_cap=any(bool(result.hit_cap) for result in results),
        )
        self.record_count += 1
        level = self._level_key()
        input_hash = self._text_hash(prompt.text)
        best_index = (
            int(np.argmax(np.asarray([item.performance_score for item in per_result_metrics], dtype=np.float64)))
            if per_result_metrics
            else None
        )
        best_result = results[best_index] if best_index is not None else None
        output_hash = self._optional_text_hash(best_result.completion_text if best_result else None)
        self.unique_input_hashes.add(input_hash)
        self.unique_output_hashes.add(output_hash)
        if bool(metrics.success):
            self.unique_success_input_hashes.add(input_hash)
            self.unique_success_output_hashes.add(output_hash)
        z_hash = self._latent_hash(latent_point)
        if z_hash is not None:
            self.unique_z_hashes.add(z_hash)
        self._store_latent_metrics(
            latent_point=latent_point,
            sample_index=sample_index,
            metrics=metrics,
        )
        self.level_eval_counts[level] = int(self.level_eval_counts.get(level, 0)) + 1
        self._level_set(self.level_input_hashes, level).add(input_hash)
        self._level_set(self.level_output_hashes, level).add(output_hash)
        output_counts = self.level_output_hash_counts.setdefault(level, Counter())
        output_counts[output_hash] += 1
        if bool(metrics.success):
            self._level_set(self.level_success_input_hashes, level).add(input_hash)
            self._level_set(self.level_success_output_hashes, level).add(output_hash)

        record = EvaluationRecord(
            sample_index=int(sample_index),
            subset_level=self.current_subset_level,
            length=int(aggregate_length),
            output_token_margin=int(aggregate_length) - int(self.threshold),
            success=bool(metrics.success),
            hit_cap=bool(metrics.hit_cap),
            elapsed_sec=float(metrics.elapsed_sec),
            tokens_per_second=float(metrics.tokens_per_second),
            input_token_count=int(best_result.input_token_count if best_result else 0),
            input_token_ids=[int(token_id) for token_id in prompt.token_ids],
            rendered_input_text=prompt.text,
            completion_text=best_result.completion_text if best_result else None,
            repeat_lengths=lengths,
            repeat_aggregation=str(self.objective_aggregation),
            repeat_count=len(lengths),
            best_repeat_index=best_index,
            performance_metric=str(metrics.performance_metric),
            performance_score=float(metrics.performance_score),
            performance_threshold=float(metrics.performance_threshold),
            performance_margin=float(metrics.performance_margin),
            repetition_score=float(metrics.repetition_score),
            rep_2=float(metrics.rep_2),
            rep_3=float(metrics.rep_3),
            rep_4=float(metrics.rep_4),
        )
        self._store_evaluation_trace(
            record,
            input_hash=input_hash,
            output_hash=output_hash,
            latent_hash=z_hash,
        )
        self._append_raw_sample(
            record=record,
            latent_point=latent_point,
            results=results,
            result_metrics=per_result_metrics,
            input_hash=input_hash,
            latent_hash=z_hash,
        )
        self._store_record(record)
        return metrics

    @torch.inference_mode()
    def evaluate_prompt_repeated(
        self,
        prompt: ProjectedPrompt,
        latent_point: Optional[np.ndarray] = None,
    ) -> GenerationMetrics:
        repeat_count = max(1, int(self.objective_repeats))
        sample_index = self._start_sample_progress()
        results = []
        for start in range(0, repeat_count, max(1, int(self.generation_batch_size))):
            count = min(max(1, int(self.generation_batch_size)), repeat_count - start)
            results.extend(self.generator.generate_batch([prompt.text] * count))
        self._update_sample_progress(
            sample_index,
            int(round(self._aggregate_repeat_lengths([r.completion_length for r in results]))),
            final=True,
        )
        return self._record_repeated_evaluation(
            sample_index=sample_index,
            prompt=prompt,
            results=results,
            latent_point=latent_point,
        )

    @torch.inference_mode()
    def evaluate_prompt(
        self,
        prompt: ProjectedPrompt,
        latent_point: Optional[np.ndarray] = None,
    ) -> GenerationMetrics:
        if max(1, int(self.objective_repeats)) > 1:
            return self.evaluate_prompt_repeated(prompt, latent_point=latent_point)
        sample_index = self._start_sample_progress()
        result = self.generator.generate(
            prompt.text,
            output_token_callback=lambda count: self._update_sample_progress(sample_index, count),
        )
        self._update_sample_progress(sample_index, int(result.completion_length), final=True)
        metrics = self._metrics_from_result(
            result=result,
            elapsed_sec=result.elapsed_sec,
            tokens_per_second=result.tokens_per_second,
            hit_cap=result.hit_cap,
        )
        self._record_evaluation(
            sample_index=sample_index,
            prompt=prompt,
            result=result,
            metrics=metrics,
            latent_point=latent_point,
        )
        return metrics

    @torch.inference_mode()
    def evaluate_prompts_batch(
        self,
        prompts: List[ProjectedPrompt],
        latent_points: Optional[np.ndarray] = None,
    ) -> List[GenerationMetrics]:
        if not prompts:
            return []
        batch_size = max(1, int(self.generation_batch_size))
        repeat_count = max(1, int(self.objective_repeats))
        if repeat_count > 1:
            return self.evaluate_prompts_repeated_batch(prompts, latent_points=latent_points)
        if batch_size == 1 or len(prompts) == 1:
            metrics: List[GenerationMetrics] = []
            for index, prompt in enumerate(prompts):
                latent_point = None
                if latent_points is not None:
                    latent_point = np.asarray(latent_points[index], dtype=np.float32)
                metrics.append(self.evaluate_prompt(prompt, latent_point=latent_point))
            return metrics

        all_metrics: List[GenerationMetrics] = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start: start + batch_size]
            sample_indices = self._reserve_sample_indices(len(chunk))
            results = self.generator.generate_batch(
                [prompt.text for prompt in chunk],
                output_token_callbacks=None,
            )
            self._write_batch_summary(
                sample_indices,
                [int(result.completion_length) for result in results],
            )
            for local_index, (prompt, result) in enumerate(zip(chunk, results)):
                sample_index = int(sample_indices[local_index])
                metrics = self._metrics_from_result(
                    result=result,
                    elapsed_sec=result.elapsed_sec,
                    tokens_per_second=result.tokens_per_second,
                    hit_cap=result.hit_cap,
                )
                latent_point = None
                if latent_points is not None:
                    latent_point = np.asarray(latent_points[start + local_index], dtype=np.float32)
                self._record_evaluation(
                    sample_index=sample_index,
                    prompt=prompt,
                    result=result,
                    metrics=metrics,
                    latent_point=latent_point,
                )
                all_metrics.append(metrics)
        return all_metrics

    @torch.inference_mode()
    def evaluate_prompts_repeated_batch(
        self,
        prompts: List[ProjectedPrompt],
        latent_points: Optional[np.ndarray] = None,
    ) -> List[GenerationMetrics]:
        if not prompts:
            return []
        repeat_count = max(1, int(self.objective_repeats))
        batch_size = max(1, int(self.generation_batch_size))
        sample_indices = self._reserve_sample_indices(len(prompts))
        flat_prompt_texts = [prompt.text for prompt in prompts for _ in range(repeat_count)]
        flat_results = []
        for start in range(0, len(flat_prompt_texts), batch_size):
            chunk = flat_prompt_texts[start: start + batch_size]
            flat_results.extend(self.generator.generate_batch(chunk))
        metrics: List[GenerationMetrics] = []
        for index, prompt in enumerate(prompts):
            start = index * repeat_count
            results = flat_results[start: start + repeat_count]
            lengths = [int(result.completion_length) for result in results]
            aggregate_length = int(round(self._aggregate_repeat_lengths(lengths)))
            self._update_sample_progress(sample_indices[index], aggregate_length, final=True)
            latent_point = None
            if latent_points is not None:
                latent_point = np.asarray(latent_points[index], dtype=np.float32)
            metrics.append(
                self._record_repeated_evaluation(
                    sample_index=sample_indices[index],
                    prompt=prompt,
                    results=results,
                    latent_point=latent_point,
                )
            )
        return metrics

    @torch.inference_mode()
    def evaluate_token_ids(
        self,
        token_ids: List[int],
        latent_point: Optional[np.ndarray] = None,
    ) -> GenerationMetrics:
        prompt = ProjectedPrompt(
            token_ids=[int(token_id) for token_id in token_ids],
            text=self.projection_space.token_ids_to_text(token_ids),
        )
        return self.evaluate_prompt(prompt, latent_point=latent_point)

    @torch.inference_mode()
    def evaluate_z(self, z: Union[np.ndarray, torch.Tensor]) -> GenerationMetrics:
        if isinstance(z, np.ndarray):
            latent_point = z.astype(np.float32, copy=False)
        else:
            latent_point = z.detach().to("cpu", torch.float32).numpy()
        return self.evaluate_prompt(self.projection_space(latent_point), latent_point=latent_point)

    @torch.inference_mode()
    def evaluate_z_batch(self, z: Union[np.ndarray, torch.Tensor]) -> List[GenerationMetrics]:
        if isinstance(z, np.ndarray):
            latent_points = z.astype(np.float32, copy=False)
        else:
            latent_points = z.detach().to("cpu", torch.float32).numpy()
        if latent_points.ndim == 1:
            latent_points = latent_points.reshape(1, -1)
        prompts = self.projection_space.decode_batch(latent_points)
        return self.evaluate_prompts_batch(prompts, latent_points=latent_points)

    @torch.inference_mode()
    def g_single(self, z: Union[np.ndarray, torch.Tensor]) -> float:
        metrics = self.evaluate_z(z)
        return float(metrics.performance_threshold - metrics.performance_score)

    @torch.inference_mode()
    def __call__(self, z: Union[np.ndarray, torch.Tensor], to_numpy: bool = False):
        if isinstance(z, np.ndarray):
            z_tensor = torch.from_numpy(z.astype(np.float32))
        else:
            z_tensor = z.to(torch.float32)
        if z_tensor.ndim == 1:
            z_tensor = z_tensor.unsqueeze(0)
        if int(self.generation_batch_size) > 1 and int(z_tensor.shape[0]) > 1:
            metrics = self.evaluate_z_batch(z_tensor)
            values = [
                float(item.performance_threshold - item.performance_score)
                for item in metrics
            ]
        else:
            values = [self.g_single(z_tensor[index]) for index in range(int(z_tensor.shape[0]))]
        if to_numpy:
            return np.asarray(values, dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32)
