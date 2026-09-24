from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from estimators_samplers import SubsetTraceObserver
from utils.subset_trace_reporting import save_subset_trace_checkpoint


@dataclass
class _Summary:
    level: int
    threshold: float


class SubsetTraceCheckpointTest(unittest.TestCase):
    def test_checkpoint_preserves_completed_population_atomically(self) -> None:
        observer = SubsetTraceObserver()
        observer.observe_population(
            level=0,
            points=np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
            values=np.asarray([-2.0, -1.0]),
            log_targets=np.asarray([-0.5, -0.5]),
            elite_count=1,
            threshold=-2.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            save_subset_trace_checkpoint(
                path=directory,
                observer=observer,
                completed_level_summaries=[_Summary(level=0, threshold=-2.0)],
            )
            checkpoint_dir = Path(directory) / "checkpoint"
            metadata = json.loads(
                (checkpoint_dir / "checkpoint.json").read_text(encoding="utf-8")
            )
            vectors = np.load(checkpoint_dir / "latent_states.npz")

        self.assertFalse(metadata["complete"])
        self.assertEqual(metadata["completed_level_count"], 1)
        self.assertEqual(vectors["latent_vectors"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
