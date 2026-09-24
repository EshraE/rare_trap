"""Non-numerical execution logging shared by experiment entry points."""
import contextlib
import _thread
import json
import logging
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import traceback

from raretrap.artifacts import DatasetJournal, atomic_json


def redact(text):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "[REDACTED]", text)


class RedactedFormatter(logging.Formatter):
    def format(self, record):
        return redact(super().format(record))


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        original_length = len(text)
        text = redact(text)
        self.file.write(text)
        self.file.flush()
        self.stream.write(text)
        return original_length

    def flush(self):
        self.file.flush()
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


class RunInterrupted(RuntimeError):
    """Termination requested; completed raw records remain on disk."""


class RunMonitor:
    """Observe the fsynced raw journal, never the sampler or its random streams."""
    def __init__(self, output, interval=2.0):
        self.output, self.interval = Path(output), interval
        self.journal = DatasetJournal(self.output / "raw_samples.jsonl", self.output / "dataset")
        try:
            self.events = (self.output / "events.jsonl").open("x", encoding="utf-8")
        except BaseException:
            self.journal.close()
            raise
        self.closed = False
        self.stop = threading.Event()
        self.error = None
        self.started = time.time()
        self.last_heartbeat = 0.0
        self.thread = threading.Thread(target=self._watch, name="raretrap-artifacts", daemon=True)

    def event(self, event, **fields):
        self.events.write(json.dumps(dict(schema_version=1, timestamp=time.time(), event=event,
                                         **fields), allow_nan=False) + "\n")
        self.events.flush()
        os.fsync(self.events.fileno())

    def checkpoint(self, state):
        manifest = self.journal.checkpoint(state)
        status = dict(manifest, started=self.started, elapsed_sec=time.time() - self.started,
                      pid=os.getpid())
        atomic_json(self.output / "status.json", status)
        return status

    def start(self):
        self.event("run_started")
        self.checkpoint("running")
        self.thread.start()

    def _watch(self):
        try:
            while not self.stop.wait(self.interval):
                added = self.journal.poll()
                self.checkpoint("running")
                now = time.time()
                if added or now - self.last_heartbeat >= 30:
                    self.event("samples_saved" if added else "heartbeat",
                               completed_evaluations=self.journal.evaluations,
                               dataset_rows=self.journal.rows, **self.journal.latest)
                    self.last_heartbeat = now
        except BaseException as exc:
            self.error = exc
            # Fail closed: do not knowingly continue expensive evaluations after
            # a disk/export error. The main thread handles cleanup at its next
            # Python signal checkpoint, including after a blocking GPU call.
            _thread.interrupt_main()

    def finish(self, state):
        if self.closed:
            return
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        try:
            if self.error:
                raise RuntimeError("Live artifact writer failed; inspect raw_samples.jsonl and disk space.") from self.error
            self.journal.poll()
            if state == "complete" and self.journal.manifest(state)["pending_raw_bytes"]:
                raise ValueError("Run ended with an incomplete raw sample record.")
            self.checkpoint(state)
            self.event("run_" + state, completed_evaluations=self.journal.evaluations,
                       dataset_rows=self.journal.rows)
        except BaseException:
            with contextlib.suppress(OSError, ValueError):
                self.checkpoint("failed")
            raise
        finally:
            self.closed = True
            try:
                self.journal.close()
            finally:
                self.events.close()


@contextlib.contextmanager
def run_log(output: Path):
    """Capture logs, stream a dataset, and record success/failure explicitly."""
    output = Path(output)
    with (output / "run.log").open("x", encoding="utf-8") as log:
        handler = logging.StreamHandler(log)
        handler.setFormatter(RedactedFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(min(previous_level, logging.INFO))
        monitor = None
        old_signal = None
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                if threading.current_thread() is threading.main_thread():
                    old_signal = signal.getsignal(signal.SIGTERM)
                    def terminate(signum, frame):
                        raise RunInterrupted("Run received SIGTERM")
                    signal.signal(signal.SIGTERM, terminate)
                monitor = RunMonitor(output)
                monitor.start()
                yield
                monitor.finish("complete")
                monitor = None
            except BaseException as exc:
                # Storage itself may be failing. Cleanup and the original error
                # must not depend on successfully writing another diagnostic.
                with contextlib.suppress(OSError):
                    traceback.print_exc()
                with contextlib.suppress(OSError):
                    (output / "FAILED").write_text(type(exc).__name__ + "\n", encoding="utf-8")
                if monitor is not None:
                    try:
                        monitor.finish("interrupted" if isinstance(exc, (KeyboardInterrupt, RunInterrupted)) else "failed")
                    except BaseException:
                        with contextlib.suppress(OSError):
                            traceback.print_exc()
                            atomic_json(output / "status.json", dict(state="failed", updated=time.time(),
                                                                     reason="artifact_writer_error"))
                raise
            finally:
                if old_signal is not None:
                    signal.signal(signal.SIGTERM, old_signal)
                root_logger.removeHandler(handler)
                root_logger.setLevel(previous_level)
                handler.close()
