from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Optional

from chess_reasoning.utils.io import read_jsonl


def aggregate_move_ranks(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        pid = row.get("puzzle_id")
        prompt = row.get("prompt_style") or "scoring_only"
        if not pid:
            continue
        grouped[(pid, prompt)].append(row)

    output = []
    for (pid, prompt), items in grouped.items():
        book_move = items[0].get("book_move")
        generated_move = items[0].get("generated_move")
        best = max(items, key=lambda r: r.get("logprob_total", float("-inf")))
        topprob_move = best.get("candidate_move")
        topprob_logprob = best.get("logprob_total")

        book_row = next((r for r in items if r.get("candidate_move") == book_move), None)
        gen_row = next((r for r in items if r.get("candidate_move") == generated_move), None)

        book_rank = book_row.get("rank_among_candidates") if book_row else None
        book_logprob = book_row.get("logprob_total") if book_row else None
        gen_logprob = gen_row.get("logprob_total") if gen_row else None

        margin = None
        if book_logprob is not None and topprob_logprob is not None:
            margin = book_logprob - topprob_logprob

        output.append(
            {
                "puzzle_id": pid,
                "prompt_style": prompt,
                "book_move": book_move,
                "generated_move": generated_move,
                "topprob_move": topprob_move,
                "book_move_rank": book_rank,
                "book_move_logprob": book_logprob,
                "topprob_logprob": topprob_logprob,
                "generated_move_logprob": gen_logprob,
                "margin_book_vs_top": margin,
                "book_move_in_top1": bool(book_rank == 1) if book_rank is not None else False,
                "book_move_in_top3": bool(book_rank is not None and book_rank <= 3),
                "book_move_in_top5": bool(book_rank is not None and book_rank <= 5),
                "generated_equals_topprob": generated_move == topprob_move if generated_move else False,
            }
        )

    return output


def summarize_move_ranks(rows: Iterable[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get("prompt_style") or "scoring_only"].append(row)

    def _as_bool(val: object) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {"true", "1", "yes"}
        return bool(val)

    summaries = []
    for prompt, items in buckets.items():
        def _as_float(v: object) -> Optional[float]:
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        ranks = [_as_float(r.get("book_move_rank")) for r in items]
        ranks = [r for r in ranks if r is not None]
        margins = [_as_float(r.get("margin_book_vs_top")) for r in items]
        margins = [m for m in margins if m is not None]
        top1 = sum(1 for r in items if _as_bool(r.get("book_move_in_top1")))
        top3 = sum(1 for r in items if _as_bool(r.get("book_move_in_top3")))
        top5 = sum(1 for r in items if _as_bool(r.get("book_move_in_top5")))
        gen_eq = sum(1 for r in items if _as_bool(r.get("generated_equals_topprob")))

        summaries.append(
            {
                "prompt_style": prompt,
                "n": len(items),
                "top1_book_rate": top1 / len(items) if items else 0.0,
                "top3_book_rate": top3 / len(items) if items else 0.0,
                "top5_book_rate": top5 / len(items) if items else 0.0,
                "mean_book_rank": mean(ranks) if ranks else None,
                "median_book_rank": median(ranks) if ranks else None,
                "mean_margin_book_vs_top": mean(margins) if margins else None,
                "mean_generated_equals_topprob": gen_eq / len(items) if items else 0.0,
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


def aggregate_from_file(input_path: str, output_path: str) -> list[dict]:
    rows = list(read_jsonl(input_path))
    aggregated = aggregate_move_ranks(rows)
    write_csv(output_path, aggregated)
    return aggregated


def move_rank_report(input_rows: list[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    summaries = summarize_move_ranks(input_rows)
    write_csv(output_dir / "tokenprob_ablation_summary.csv", summaries)
