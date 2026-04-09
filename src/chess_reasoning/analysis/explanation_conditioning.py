from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from chess_reasoning.utils.io import read_jsonl


def _load_logprobs(path: str) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        pid = row.get("puzzle_id")
        prompt = row.get("prompt_style") or "scoring_only"
        if not pid or not prompt:
            continue
        grouped[(pid, prompt)].append(row)
    return grouped


def _as_float(val: object) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def compute_explanation_delta(
    base_logprobs_path: str,
    expl_logprobs_path: str,
) -> list[dict]:
    base = _load_logprobs(base_logprobs_path)
    expl = _load_logprobs(expl_logprobs_path)

    rows = []
    for (pid, prompt), items in expl.items():
        base_items = base.get((pid, "scoring_only")) or base.get((pid, prompt))
        if not base_items:
            continue
        book_move = items[0].get("book_move")
        # find book row
        book_row = next((r for r in items if r.get("candidate_move") == book_move), None)
        base_row = next((r for r in base_items if r.get("candidate_move") == book_move), None)
        if not book_row or not base_row:
            continue
        book_rank = book_row.get("rank_among_candidates")
        try:
            book_rank = int(book_rank) if book_rank is not None else None
        except Exception:
            book_rank = None
        book_logprob = _as_float(book_row.get("logprob_total"))
        base_logprob = _as_float(base_row.get("logprob_total"))
        delta = None
        if book_logprob is not None and base_logprob is not None:
            delta = book_logprob - base_logprob

        rows.append(
            {
                "puzzle_id": pid,
                "prompt_style": prompt,
                "book_move": book_move,
                "book_move_rank": book_rank,
                "book_move_logprob_expl": book_logprob,
                "book_move_logprob_base": base_logprob,
                "delta_expl": delta,
                "book_move_in_top1": book_rank == 1 if book_rank is not None else False,
                "book_move_in_top3": book_rank is not None and book_rank <= 3,
                "mrr_book": 1.0 / book_rank if book_rank else 0.0,
            }
        )
    return rows


def summarize_explanation_delta(rows: Iterable[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("prompt_style")].append(row)

    summaries = []
    for prompt, items in buckets.items():
        deltas = [r.get("delta_expl") for r in items if r.get("delta_expl") is not None]
        ranks = [r.get("book_move_rank") for r in items if r.get("book_move_rank") is not None]
        mrrs = [r.get("mrr_book") for r in items if r.get("mrr_book") is not None]
        top1 = sum(1 for r in items if r.get("book_move_in_top1"))
        top3 = sum(1 for r in items if r.get("book_move_in_top3"))

        summaries.append(
            {
                "prompt_style": prompt,
                "n": len(items),
                "top1_book_rate": top1 / len(items) if items else 0.0,
                "top3_book_rate": top3 / len(items) if items else 0.0,
                "mrr_book": mean(mrrs) if mrrs else None,
                "mean_delta_expl": mean(deltas) if deltas else None,
                "median_delta_expl": median(deltas) if deltas else None,
                "mean_book_rank": mean(ranks) if ranks else None,
            }
        )
    return summaries


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
