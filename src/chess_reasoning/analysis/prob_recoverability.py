from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from chess_reasoning.utils.io import read_jsonl


def _as_float(val: object) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except Exception:
        return None


def _as_int(val: object) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(float(val))
    except Exception:
        return None


def _as_bool(val: object) -> bool | None:
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "1", "yes"}
    return bool(val)


def load_rank_table(path: str) -> dict[tuple[str, str], dict]:
    data: dict[tuple[str, str], dict] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("puzzle_id")
            prompt = row.get("prompt_style")
            if not pid or not prompt:
                continue
            data[(pid, prompt)] = {
                "book_move_rank": _as_int(row.get("book_move_rank")),
                "margin_book_vs_top": _as_float(row.get("margin_book_vs_top")),
            }
    return data


def load_recoverability(path: str) -> dict[tuple[str, str], dict]:
    data: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(path):
        pid = row.get("puzzle_id")
        prompt = row.get("prompt_condition") or row.get("prompt_style")
        if not pid or not prompt:
            continue
        recoverable = row.get("recoverable_exact_strict")
        if recoverable is None:
            recoverable = row.get("recoverable_exact")
        recoverable = _as_bool(recoverable)
        data[(pid, prompt)] = {
            "recoverable": recoverable,
            "engine_delta": _as_float(row.get("engine_delta_cp")),
        }
    return data


def build_merged_table(rank_path: str, recoverability_path: str) -> list[dict]:
    ranks = load_rank_table(rank_path)
    rec = load_recoverability(recoverability_path)

    merged = []
    for key, rvals in ranks.items():
        if key not in rec:
            continue
        pid, prompt = key
        merged.append(
            {
                "puzzle_id": pid,
                "prompt_style": prompt,
                "book_move_rank": rvals.get("book_move_rank"),
                "margin_book_vs_top": rvals.get("margin_book_vs_top"),
                "recoverability": rec[key].get("recoverable"),
                "engine_delta": rec[key].get("engine_delta"),
            }
        )
    return merged


def _bucket_rank(rank: int | None) -> str | None:
    if rank is None:
        return None
    if rank == 1:
        return "1"
    if 2 <= rank <= 3:
        return "2-3"
    if 4 <= rank <= 5:
        return "4-5"
    return ">5"


def _bucket_margin(margin: float | None) -> str | None:
    if margin is None:
        return None
    if margin >= -0.5:
        return ">=-0.5"
    if margin >= -2.0:
        return "-2 to -0.5"
    if margin >= -5.0:
        return "-5 to -2"
    return "< -5"


def summarize_by_bucket(rows: Iterable[dict], bucket_field: str) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row.get(bucket_field)
        if key is None:
            continue
        buckets[str(key)].append(row)

    summaries = []
    for key, items in buckets.items():
        rec_vals = [r.get("recoverability") for r in items if r.get("recoverability") is not None]
        deltas = [r.get("engine_delta") for r in items if r.get("engine_delta") is not None]
        rec_rate = sum(1 for r in rec_vals if r) / len(rec_vals) if rec_vals else 0.0
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        summaries.append(
            {
                "bucket": key,
                "n": len(items),
                "recoverability_rate": rec_rate,
                "mean_engine_delta": mean_delta,
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


def compute_prob_recoverability(
    rank_path: str,
    recoverability_path: str,
    merged_output: str,
    rank_bucket_output: str,
    margin_bucket_output: str,
) -> None:
    merged = build_merged_table(rank_path, recoverability_path)
    # add buckets
    for row in merged:
        row["rank_bucket"] = _bucket_rank(row.get("book_move_rank"))
        row["margin_bucket"] = _bucket_margin(row.get("margin_book_vs_top"))

    write_csv(merged_output, merged)
    rank_summary = summarize_by_bucket(merged, "rank_bucket")
    margin_summary = summarize_by_bucket(merged, "margin_bucket")
    write_csv(rank_bucket_output, rank_summary)
    write_csv(margin_bucket_output, margin_summary)
