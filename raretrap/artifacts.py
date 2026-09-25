"""Durable, streaming operational artifacts, separate from numerical routines."""
from __future__ import annotations

import hashlib
import contextlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from raretrap.terminology import projection_name


MAX_JSONL_LINE_BYTES = 64 * 1024 * 1024


def jsonl_records(path: Path, *, committed_only: bool = False, byte_limit: int | None = None):
    """Read a bounded point-in-time prefix, without loading a whole raw dataset.

    Raw evaluation journals commit records with a newline. Other completed JSONL
    exports may omit the last newline. Complete malformed records always fail.
    """
    if byte_limit is not None and (type(byte_limit) is not int or byte_limit < 0):
        raise ValueError("JSONL snapshot size must be a nonnegative integer.")
    with Path(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if byte_limit is not None:
            if size < byte_limit:
                raise ValueError("JSONL source was truncated below its snapshot size.")
            size = byte_limit
        while stream.tell() < size:
            line = stream.readline(min(size - stream.tell(), MAX_JSONL_LINE_BYTES + 1))
            if len(line) > MAX_JSONL_LINE_BYTES:
                raise ValueError("JSONL record exceeds the supported 64 MiB line size.")
            if not line:
                raise ValueError("JSONL source was truncated during reading.")
            if committed_only and not line.endswith(b"\n"):
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL records must be objects.")
            yield row
        if os.fstat(stream.fileno()).st_size < size:
            raise ValueError("JSONL source was truncated during reading.")


def atomic_json(path: Path, value: dict) -> None:
    """Readers see either the previous complete document or its replacement."""
    path = Path(path)
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def dataset_rows(row: dict):
    """One schema across projection arms and estimator evaluations/repeats."""
    if row.get("record_type") == "metadata":
        return
    estimator = row.get("record_type") == "sample"
    if not estimator and "arm" not in row:
        raise ValueError("Unknown raw sample schema; refusing an ambiguous dataset export.")
    arm = None if estimator else projection_name(row["arm"])
    for index, generation in enumerate(row["generations"] if estimator else [row]):
        metrics = {}
        invalid = False
        for key in ("output_tokens", "repetition_score", "rep_2", "rep_3", "rep_4",
                    "elapsed_sec", "tokens_per_second", "performance_score",
                    "performance_threshold", "performance_margin"):
            value = generation.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                value, invalid = None, True
            metrics[key] = value
        yield dict(schema_version=1, experiment="estimate" if estimator else "projection",
                   sample_index=row["sample_index"], repeat_index=generation.get("repeat_index", index),
                   subset_level=row.get("subset_level"), arm=arm,
                   prompt_text=row.get("input_text", row.get("prompt_text")),
                   surrogate_token_ids=row.get("input_token_ids", row.get("surrogate_token_ids")),
                   latent_vector_float32=row.get("latent_vector_float32"),
                   model_input_token_ids=generation.get("model_input_token_ids"),
                   completion_token_ids=generation.get("completion_token_ids"),
                   completion_text=generation.get("completion_text"), metrics=metrics,
                   success=generation.get("success"), hit_cap=generation.get("hit_cap"),
                   invalid_metrics=invalid)


class DatasetJournal:
    """Tail committed raw lines without holding the growing dataset in memory.

    The raw stream is authoritative. A derived dataset can always be rebuilt;
    an unterminated final raw line is never interpreted as a completed evaluation.
    """
    MAX_LINE_BYTES = MAX_JSONL_LINE_BYTES

    def __init__(self, source: Path, destination: Path, *, byte_limit: int | None = None):
        if byte_limit is not None and (type(byte_limit) is not int or byte_limit < 0):
            raise ValueError("Raw snapshot size must be a nonnegative integer.")
        self.source = Path(source)
        self.destination = Path(destination)
        self.destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.handle = (self.destination / "samples.jsonl").open("x", encoding="utf-8", newline="\n")
        self.byte_limit = byte_limit
        self.offset = self.evaluations = self.rows = 0
        self.digest = hashlib.sha256()
        self.latest = {}
        self._source_identity = None
        self._observed_size = 0

    def _check_source(self, info) -> int:
        identity = (info.st_dev, info.st_ino)
        if self._source_identity is not None and identity != self._source_identity:
            raise ValueError("Raw stream was replaced while being monitored.")
        if info.st_size < max(self.offset, self._observed_size, self.byte_limit or 0):
            raise ValueError("Raw stream was truncated while being monitored.")
        self._source_identity = identity
        self._observed_size = info.st_size
        return min(info.st_size, self.byte_limit) if self.byte_limit is not None else info.st_size

    def _missing_source(self) -> None:
        if self._source_identity is not None or self.byte_limit is not None:
            raise ValueError("Raw stream disappeared while being monitored.")

    def poll(self) -> int:
        previous = self.evaluations
        try:
            stream = self.source.open("rb")
        except FileNotFoundError:
            self._missing_source()
            return 0
        with stream:
            size = self._check_source(os.fstat(stream.fileno()))
            stream.seek(self.offset)
            while stream.tell() < size:
                start = stream.tell()
                line = stream.readline(min(size - start, self.MAX_LINE_BYTES + 1))
                if len(line) > self.MAX_LINE_BYTES:
                    raise ValueError("Raw sample exceeds the supported 64 MiB line size.")
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Raw sample records must be objects.")
                for normalized in dataset_rows(row):
                    encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False) + "\n"
                    self.handle.write(encoded)
                    self.digest.update(encoded.encode("utf-8"))
                    self.rows += 1
                if row.get("record_type") != "metadata":
                    self.evaluations += 1
                    metric = row.get("aggregate", row)
                    self.latest = dict(sample_index=row["sample_index"], subset_level=row.get("subset_level"),
                                       arm=row.get("arm"), output_tokens=metric.get("output_tokens"),
                                       repetition_score=metric.get("repetition_score"))
                    for key, value in self.latest.items():
                        if isinstance(value, float) and not math.isfinite(value):
                            self.latest[key] = None
                self.offset = stream.tell()
            self._check_source(os.fstat(stream.fileno()))
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self.evaluations - previous

    def manifest(self, state: str) -> dict:
        try:
            size = self._check_source(self.source.stat())
        except FileNotFoundError:
            self._missing_source()
            size = 0
        return dict(schema_version=1, state=state, updated=time.time(), format="jsonl",
                    completed_evaluations=self.evaluations, dataset_rows=self.rows,
                    raw_bytes_consumed=self.offset, pending_raw_bytes=size - self.offset,
                    dataset_sha256=self.digest.hexdigest(), latest=self.latest,
                    semantics="Evaluated candidates, not retained MCMC population slots. No importance weights.")

    def checkpoint(self, state: str) -> dict:
        manifest = self.manifest(state)
        atomic_json(self.destination / "manifest.json", manifest)
        return manifest

    def close(self):
        self.handle.close()


def export_dataset(source: Path, destination: Path) -> dict:
    """Read-only point-in-time recovery/export; never modify the source stream."""
    source = Path(source)
    if not source.is_file():
        raise ValueError("Input run has no raw_samples.jsonl file.")
    journal = DatasetJournal(source, destination, byte_limit=source.stat().st_size)
    try:
        journal.poll()
        # A snapshot is not a claim that the experiment completed successfully.
        return journal.checkpoint("snapshot")
    except BaseException:
        # A full disk must not hide the original failure behind a diagnostic error.
        with contextlib.suppress(OSError, ValueError):
            journal.checkpoint("failed")
        raise
    finally:
        journal.close()
