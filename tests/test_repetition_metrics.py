from __future__ import annotations

import unittest

from utils.repetition_metrics import repetition_metrics


class RepetitionMetricsTest(unittest.TestCase):
    def test_unique_sequence_has_zero_repetition(self) -> None:
        metrics = repetition_metrics([1, 2, 3, 4, 5])

        self.assertEqual(metrics.rep_2, 0.0)
        self.assertEqual(metrics.rep_3, 0.0)
        self.assertEqual(metrics.rep_4, 0.0)
        self.assertEqual(metrics.repetition_score, 0.0)

    def test_repeated_sequence_uses_inverted_multingram_diversity(self) -> None:
        metrics = repetition_metrics([7, 7, 7, 7, 7])

        self.assertEqual(metrics.rep_2, 0.75)
        self.assertAlmostEqual(metrics.rep_3, 2.0 / 3.0)
        self.assertEqual(metrics.rep_4, 0.5)
        self.assertAlmostEqual(metrics.repetition_score, 1.0 - (0.25 * (1.0 / 3.0) * 0.5))

    def test_short_sequences_and_special_tokens_do_not_divide_by_zero(self) -> None:
        metrics = repetition_metrics([0, 2, 0], special_token_ids=[0])

        self.assertEqual(metrics.rep_2, 0.0)
        self.assertEqual(metrics.rep_3, 0.0)
        self.assertEqual(metrics.rep_4, 0.0)
        self.assertEqual(metrics.repetition_score, 0.0)


if __name__ == "__main__":
    unittest.main()
