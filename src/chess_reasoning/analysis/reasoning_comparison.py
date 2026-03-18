from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Iterable, Iterator, Optional

from chess_reasoning.evaluation.move_quality import exact_best_move, is_legal_move


def extract_reasoning_text(row: dict) -> str:
    for key in (
        "reasoning_text",
        "parsed_explanation",
        "clean_text",
        "raw_text",
        "llm_explanation",
        "transcript_clean",
        "transcript_raw",
        "raw_response",
    ):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def _base_puzzle_fields(puzzles: dict[str, dict], puzzle_id: str | None) -> dict:
    puzzle = puzzles.get(puzzle_id) if puzzle_id else None
    return {
        "puzzle_id": puzzle_id,
        "fen": puzzle.get("fen") if puzzle else None,
        "section": puzzle.get("section") if puzzle else None,
        "book_move": puzzle.get("best_move") if puzzle else None,
        "source": puzzle.get("source") if puzzle else None,
    }


def _compute_correct_and_legal(
    fen: Optional[str],
    best_move: Optional[str],
    move_uci: Optional[str],
) -> tuple[Optional[bool], Optional[bool]]:
    if not move_uci:
        return None, None
    legal = None
    if fen:
        legal = is_legal_move(fen, move_uci)
    correct = None
    if best_move:
        correct = exact_best_move(best_move, move_uci)
    return correct, legal


def build_reasoning_table(
    puzzles: dict[str, dict],
    llm_rows: Iterable[dict] = (),
    human_rows: Iterable[dict] = (),
    book_rows: Iterable[dict] = (),
) -> Iterator[dict]:
    for row in llm_rows:
        pid = row.get("puzzle_id")
        base = _base_puzzle_fields(puzzles, pid)
        move_uci = row.get("parsed_move") or row.get("move_uci") or row.get("chosen_move")
        fen = base.get("fen")
        best_move = base.get("book_move")
        correct = row.get("correct_move")
        legal = row.get("legal_move")
        if correct is None or legal is None:
            calc_correct, calc_legal = _compute_correct_and_legal(fen, best_move, move_uci)
            if correct is None:
                correct = calc_correct
            if legal is None:
                legal = calc_legal

        yield {
            **base,
            "source_type": "llm",
            "source_id": row.get("generation_id"),
            "model_name": row.get("model_name"),
            "provider": row.get("provider"),
            "prompt_condition": row.get("prompt_condition"),
            "participant_id": None,
            "move_uci": move_uci,
            "move_correct": correct,
            "legal_move": legal,
            "engine_delta_cp": row.get("engine_delta_cp"),
            "engine_best_move": row.get("engine_best_move"),
            "reasoning_text": extract_reasoning_text(row),
        }

    for row in human_rows:
        pid = row.get("puzzle_id")
        base = _base_puzzle_fields(puzzles, pid)
        move_uci = row.get("chosen_move") or row.get("move_uci")
        fen = base.get("fen") or row.get("fen")
        best_move = base.get("book_move")
        correct, legal = _compute_correct_and_legal(fen, best_move, move_uci)
        yield {
            **base,
            "source_type": "human",
            "source_id": row.get("explanation_id") or row.get("response_id"),
            "model_name": None,
            "provider": None,
            "prompt_condition": None,
            "participant_id": row.get("author") or row.get("metadata", {}).get("participant_id"),
            "move_uci": move_uci,
            "move_correct": correct,
            "legal_move": legal,
            "engine_delta_cp": row.get("engine_delta_cp"),
            "engine_best_move": row.get("engine_best_move"),
            "reasoning_text": extract_reasoning_text(row),
            "confidence": row.get("confidence"),
            "skill_level": row.get("skill_level"),
            "time_seconds": (row.get("metadata") or {}).get("time_seconds"),
        }

    for row in book_rows:
        pid = row.get("puzzle_id")
        base = _base_puzzle_fields(puzzles, pid)
        move_uci = row.get("parsed_move") or row.get("move_uci") or row.get("chosen_move")
        fen = base.get("fen")
        best_move = base.get("book_move")
        correct = row.get("correct_move")
        legal = row.get("legal_move")
        if correct is None or legal is None:
            calc_correct, calc_legal = _compute_correct_and_legal(fen, best_move, move_uci)
            if correct is None:
                correct = calc_correct
            if legal is None:
                legal = calc_legal

        yield {
            **base,
            "source_type": "book",
            "source_id": row.get("generation_id"),
            "model_name": row.get("model_name"),
            "provider": row.get("provider"),
            "prompt_condition": row.get("prompt_condition"),
            "participant_id": None,
            "move_uci": move_uci,
            "move_correct": correct,
            "legal_move": legal,
            "engine_delta_cp": row.get("engine_delta_cp"),
            "engine_best_move": row.get("engine_best_move"),
            "reasoning_text": extract_reasoning_text(row),
        }


def _recoverability_value(row: dict, mode: str) -> Optional[bool]:
    key = f"recoverable_exact_{mode}"
    if key in row:
        return row.get(key)
    if row.get("mask_mode") == mode:
        return row.get("recoverable_exact")
    return None


def summarize_by_source(rows: Iterable[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        group = row.get("source_type") or "unknown"
        buckets.setdefault(group, []).append(row)

    summary = []
    for group, items in buckets.items():
        summary.append(
            {
                "source_type": group,
                "n": len(items),
                "mean_engine_delta_cp": _mean(r.get("engine_delta_cp") for r in items),
                "mean_specificity": _mean(
                    r.get("specificity_score_norm", r.get("specificity_score")) for r in items
                ),
                "mean_recoverability_light": _mean(
                    1.0 if _recoverability_value(r, "light") else 0.0
                    for r in items
                    if _recoverability_value(r, "light") is not None
                ),
                "mean_recoverability_strict": _mean(
                    1.0 if _recoverability_value(r, "strict") else 0.0
                    for r in items
                    if _recoverability_value(r, "strict") is not None
                ),
                "mean_length": _mean(
                    r.get("word_count") if r.get("word_count") is not None else len((r.get("reasoning_text") or "").split())
                    for r in items
                ),
            }
        )
    return summary


def build_per_puzzle_comparison(rows: Iterable[dict]) -> list[dict]:
    by_puzzle: dict[str, dict] = {}
    for row in rows:
        pid = row.get("puzzle_id")
        if not pid:
            continue
        entry = by_puzzle.setdefault(pid, {"puzzle_id": pid, "book_move": row.get("book_move")})
        source_type = row.get("source_type")
        if source_type == "llm":
            entry.update(
                {
                    "llm_move": row.get("move_uci"),
                    "llm_delta": row.get("engine_delta_cp"),
                    "llm_specificity": row.get("specificity_score_norm", row.get("specificity_score")),
                    "llm_recoverability": _recoverability_value(row, "strict")
                    if _recoverability_value(row, "strict") is not None
                    else _recoverability_value(row, "light"),
                }
            )
        elif source_type == "human":
            entry.update(
                {
                    "human_move": row.get("move_uci"),
                    "human_delta": row.get("engine_delta_cp"),
                    "human_specificity": row.get("specificity_score_norm", row.get("specificity_score")),
                    "human_recoverability": _recoverability_value(row, "strict")
                    if _recoverability_value(row, "strict") is not None
                    else _recoverability_value(row, "light"),
                }
            )
    return list(by_puzzle.values())


def build_explanation_features(rows: Iterable[dict]) -> list[dict]:
    features = []
    for row in rows:
        features.append(
            {
                "puzzle_id": row.get("puzzle_id"),
                "source_type": row.get("source_type"),
                "move_correct": row.get("move_correct"),
                "engine_delta_cp": row.get("engine_delta_cp"),
                "square_mentions": row.get("square_mentions"),
                "move_mentions": row.get("move_mentions"),
                "tactical_keyword_count": row.get("tactical_keyword_count"),
                "generic_phrase_count": row.get("generic_phrase_count"),
                "line_depth": row.get("line_depth"),
                "specificity_score": row.get("specificity_score"),
            }
        )
    return features


def _write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_reasoning_reports(rows: Iterable[dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    summary = summarize_by_source(rows)
    per_puzzle = build_per_puzzle_comparison(rows)
    features = build_explanation_features(rows)

    _write_csv(output_dir / "reasoning_summary_by_source.csv", summary)
    _write_csv(output_dir / "per_puzzle_reasoning_comparison.csv", per_puzzle)
    _write_csv(output_dir / "explanation_features.csv", features)
