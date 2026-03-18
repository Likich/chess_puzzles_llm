from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable


def summarize_by_section(
    puzzles: dict[str, dict],
    generations: Iterable[dict],
) -> dict:
    buckets = defaultdict(list)
    for gen in generations:
        pid = gen.get("puzzle_id")
        if not pid or pid not in puzzles:
            continue
        section = puzzles[pid].get("section") or "unknown"
        buckets[section].append(gen)

    results = {}
    for section, rows in buckets.items():
        total = len(rows)
        legal = [r for r in rows if r.get("legal_move") is True]
        correct = [r for r in rows if r.get("correct_move") is True]
        deltas = [r.get("engine_delta_cp") for r in rows if r.get("engine_delta_cp") is not None]

        results[section] = {
            "total": total,
            "legal_rate": len(legal) / total if total else 0.0,
            "exact_match_rate": len(correct) / total if total else 0.0,
            "avg_engine_delta_cp": mean(deltas) if deltas else None,
        }

    return results
