from __future__ import annotations

import random
from typing import Iterable, Iterator, Optional

from chess_reasoning.utils.io import read_jsonl


def iter_filtered_puzzles(
    input_path: str,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    split: Optional[str] = None,
) -> Iterator[dict]:
    for row in read_jsonl(input_path):
        rating = row.get("rating")
        if (min_rating is not None or max_rating is not None) and rating is None:
            continue
        if min_rating is not None and rating is not None and rating < min_rating:
            continue
        if max_rating is not None and rating is not None and rating > max_rating:
            continue
        if split is not None and row.get("split") != split:
            continue
        yield row


def reservoir_sample(
    rows: Iterable[dict],
    k: int,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    n = 0
    for row in rows:
        n += 1
        if len(sample) < k:
            sample.append(row)
        else:
            j = rng.randint(0, n - 1)
            if j < k:
                sample[j] = row
    return sample
