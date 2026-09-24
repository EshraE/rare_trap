"""Token n-gram repetition metrics for generated completions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RepetitionMetrics:
    repetition_score: float
    rep_2: float
    rep_3: float
    rep_4: float


def _ngram_repetition(token_ids: list[int], n: int) -> tuple[float, float]:
    total = len(token_ids) - int(n) + 1
    if total <= 0:
        return 0.0, 1.0
    unique = len({tuple(token_ids[index: index + int(n)]) for index in range(total)})
    diversity = float(unique) / float(total)
    return 1.0 - diversity, diversity


def repetition_metrics(
    token_ids: Iterable[int],
    *,
    special_token_ids: Iterable[int] = (),
) -> RepetitionMetrics:
    special = {int(token_id) for token_id in special_token_ids}
    filtered = [int(token_id) for token_id in token_ids if int(token_id) not in special]
    rep_2, div_2 = _ngram_repetition(filtered, 2)
    rep_3, div_3 = _ngram_repetition(filtered, 3)
    rep_4, div_4 = _ngram_repetition(filtered, 4)
    score = 1.0 - (float(div_2) * float(div_3) * float(div_4))
    return RepetitionMetrics(
        repetition_score=float(score),
        rep_2=float(rep_2),
        rep_3=float(rep_3),
        rep_4=float(rep_4),
    )
