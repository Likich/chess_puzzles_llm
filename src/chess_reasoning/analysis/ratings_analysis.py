from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_ratings(
    ratings: Iterable[dict],
    items: dict[str, dict],
) -> dict:
    by_group = defaultdict(list)
    by_group_dim = defaultdict(lambda: defaultdict(list))

    for r in ratings:
        item_id = r.get("item_id")
        if not item_id or item_id not in items:
            continue
        item = items[item_id]
        group = item.get("source_group") or item.get("metadata", {}).get("source_group_hidden")
        if not group:
            continue

        overall = r.get("overall")
        if overall is not None:
            by_group[group].append(float(overall))

        ratings_map = r.get("ratings") or {}
        for dim, score in ratings_map.items():
            by_group_dim[group][dim].append(float(score))

    results = {
        "overall_by_group": {g: mean(v) for g, v in by_group.items() if v},
        "dimensions_by_group": {
            g: {dim: mean(scores) for dim, scores in dims.items() if scores}
            for g, dims in by_group_dim.items()
        },
    }
    return results
