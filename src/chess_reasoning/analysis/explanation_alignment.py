from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

from chess_reasoning.utils.io import read_jsonl


def _load_generations(path: str) -> dict[tuple[str, str], dict]:
    data = {}
    for row in read_jsonl(path):
        pid = row.get("puzzle_id")
        prompt = row.get("prompt_condition") or "scoring_only"
        if not pid:
            continue
        data[(pid, prompt)] = row
    return data


def _topprob_by_prompt(logprob_path: str) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(logprob_path):
        pid = row.get("puzzle_id")
        prompt = row.get("prompt_style") or "scoring_only"
        if not pid:
            continue
        grouped[(pid, prompt)].append(row)

    topprob = {}
    for key, items in grouped.items():
        best = max(items, key=lambda r: r.get("logprob_total", float("-inf")))
        topprob[key] = best.get("candidate_move")
    return topprob


def _recoverability_predictions(path: Optional[str]) -> dict[tuple[str, str], str]:
    if not path:
        return {}
    preds = {}
    for row in read_jsonl(path):
        pid = row.get("puzzle_id")
        prompt = row.get("prompt_condition") or "scoring_only"
        pred = row.get("predicted_move")
        if pid and pred:
            preds[(pid, prompt)] = pred
    return preds


def build_alignment_table(
    generations_path: str,
    logprob_path: str,
    recoverability_path: Optional[str],
) -> list[dict]:
    generations = _load_generations(generations_path)
    topprob = _topprob_by_prompt(logprob_path)
    preds = _recoverability_predictions(recoverability_path)

    rows = []
    for (pid, prompt), gen in generations.items():
        book_move = gen.get("book_move") or gen.get("best_move")
        generated_move = gen.get("parsed_move") or gen.get("move_uci") or gen.get("chosen_move")
        topprob_move = topprob.get((pid, prompt))
        predicted_move = preds.get((pid, prompt))

        rows.append(
            {
                "puzzle_id": pid,
                "prompt_style": prompt,
                "book_move": book_move,
                "generated_move": generated_move,
                "topprob_move": topprob_move,
                "recoverability_predicted_move": predicted_move,
                "explanation_matches_generated": predicted_move == generated_move if predicted_move else False,
                "explanation_matches_topprob": predicted_move == topprob_move if predicted_move else False,
                "explanation_matches_book": predicted_move == book_move if predicted_move else False,
                "generated_matches_topprob": generated_move == topprob_move if generated_move else False,
                "generated_matches_book": generated_move == book_move if generated_move else False,
                "topprob_matches_book": topprob_move == book_move if topprob_move else False,
            }
        )
    return rows


def summarize_alignment(rows: Iterable[dict]) -> list[dict]:
    buckets = Counter()
    for row in rows:
        generated = row.get("generated_move")
        topprob = row.get("topprob_move")
        book = row.get("book_move")
        pred = row.get("recoverability_predicted_move")

        if generated == topprob == book:
            buckets["generated = top-prob = book"] += 1
        elif generated != topprob and topprob == book:
            buckets["generated != top-prob, top-prob = book"] += 1
        elif pred == generated and generated != book:
            buckets["explanation decodes to generated but generated != book"] += 1
        elif pred and pred not in {generated, book}:
            buckets["explanation decodes to neither generated nor book"] += 1
        elif pred == topprob and pred != generated:
            buckets["explanation decodes to top-prob but not generated"] += 1

    return [{"category": k, "count": v} for k, v in buckets.items()]


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
