from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import chess

from chess_reasoning.schema import RatingItem, HumanRating
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DIMENSIONS = [
    "clarity",
    "specificity",
    "coherence",
    "helpfulness",
    "groundedness",
    "genericity_rev",
    "convincingness",
]


def _extract_text(exp: dict) -> Optional[str]:
    return exp.get("clean_text") or exp.get("parsed_explanation") or exp.get("raw_text") or exp.get("explanation_text")


def _extract_move(exp: dict) -> Optional[str]:
    return exp.get("chosen_move") or exp.get("parsed_move") or exp.get("move_uci")


def _extract_source_group(exp: dict) -> Optional[str]:
    sg = exp.get("source_group")
    if sg:
        return str(sg)
    st = exp.get("source_type")
    if st:
        if "llm" in str(st).lower() or "model" in str(st).lower():
            return "llm"
        if "human" in str(st).lower() or "stack" in str(st).lower() or "pgn" in str(st).lower():
            return "human"
    return None


def _move_san(fen: str, move_uci: str) -> Optional[str]:
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            return board.san(move)
    except Exception:
        return None
    return None


def build_rating_items(
    puzzles: Iterable[dict],
    explanations: Iterable[dict],
    design: str = "single",
    same_move: bool = True,
    per_puzzle: int = 2,
    max_puzzles: Optional[int] = None,
    seed: int = 42,
    blind: bool = True,
) -> list[dict]:
    rng = random.Random(seed)
    puzzle_map = {p["puzzle_id"]: p for p in puzzles if p.get("puzzle_id")}

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for exp in explanations:
        pid = exp.get("puzzle_id")
        if not pid or pid not in puzzle_map:
            continue
        text = _extract_text(exp)
        move = _extract_move(exp)
        if not text or not move:
            continue
        source_group = _extract_source_group(exp)
        if not source_group:
            continue
        exp = dict(exp)
        exp["_text"] = text
        exp["_move"] = move
        exp["_source_group"] = source_group
        grouped[pid][source_group].append(exp)

    puzzle_ids = list(grouped.keys())
    rng.shuffle(puzzle_ids)
    if max_puzzles is not None:
        puzzle_ids = puzzle_ids[:max_puzzles]

    items: list[dict] = []

    for pid in puzzle_ids:
        puzzle = puzzle_map[pid]
        fen = puzzle.get("fen")
        if not fen:
            continue

        group_map = grouped[pid]

        if design == "single":
            # Balanced: one per group if possible
            candidates: list[dict] = []
            if per_puzzle >= 2 and "human" in group_map and "llm" in group_map:
                if same_move:
                    human_by_move = defaultdict(list)
                    for e in group_map["human"]:
                        human_by_move[e["_move"]].append(e)
                    llm_by_move = defaultdict(list)
                    for e in group_map["llm"]:
                        llm_by_move[e["_move"]].append(e)
                    overlap_moves = [m for m in human_by_move if m in llm_by_move]
                    if not overlap_moves:
                        continue
                    move = rng.choice(overlap_moves)
                    candidates.append(rng.choice(human_by_move[move]))
                    candidates.append(rng.choice(llm_by_move[move]))
                else:
                    candidates.append(rng.choice(group_map["human"]))
                    candidates.append(rng.choice(group_map["llm"]))
            else:
                # Fallback: sample from any group
                flat = [e for g in group_map.values() for e in g]
                if not flat:
                    continue
                candidates = rng.sample(flat, k=min(per_puzzle, len(flat)))

            for exp in candidates:
                item = _make_rating_item(exp, pid, fen, blind)
                items.append(asdict(item))

        elif design == "pairwise":
            if "human" not in group_map or "llm" not in group_map:
                continue
            if same_move:
                human_by_move = defaultdict(list)
                for e in group_map["human"]:
                    human_by_move[e["_move"]].append(e)
                llm_by_move = defaultdict(list)
                for e in group_map["llm"]:
                    llm_by_move[e["_move"]].append(e)
                overlap_moves = [m for m in human_by_move if m in llm_by_move]
                if not overlap_moves:
                    continue
                move = rng.choice(overlap_moves)
                human_exp = rng.choice(human_by_move[move])
                llm_exp = rng.choice(llm_by_move[move])
            else:
                human_exp = rng.choice(group_map["human"])
                llm_exp = rng.choice(group_map["llm"])

            pair = _make_pair_item(human_exp, llm_exp, pid, fen, blind, rng)
            items.append(pair)
        else:
            raise ValueError("design must be 'single' or 'pairwise'")

    return items


def _make_rating_item(exp: dict, pid: str, fen: str, blind: bool) -> RatingItem:
    move = exp["_move"]
    move_san = _move_san(fen, move)
    source_group = exp["_source_group"]
    source_group_out = None if blind else source_group

    metadata = {
        "source_group_hidden": source_group,
        "source_type": exp.get("source_type"),
        "source_url": exp.get("source_url"),
    }

    return RatingItem(
        item_id=new_id("ritem"),
        puzzle_id=pid,
        fen=fen,
        move_uci=move,
        move_san=move_san,
        explanation_text=exp["_text"],
        source_group=source_group_out,
        source_id=exp.get("explanation_id") or exp.get("generation_id"),
        prompt_condition=exp.get("prompt_condition"),
        model_name=exp.get("model_name"),
        metadata=metadata,
    )


def _make_pair_item(human_exp: dict, llm_exp: dict, pid: str, fen: str, blind: bool, rng: random.Random) -> dict:
    # pairwise item stores two explanations; order randomized
    a_is_human = rng.choice([True, False])
    a = human_exp if a_is_human else llm_exp
    b = llm_exp if a_is_human else human_exp

    item = {
        "item_id": new_id("pair"),
        "puzzle_id": pid,
        "fen": fen,
        "move_uci": human_exp["_move"],
        "move_san": _move_san(fen, human_exp["_move"]),
        "explanation_a": a["_text"],
        "explanation_b": b["_text"],
        "source_group_a": None if blind else a["_source_group"],
        "source_group_b": None if blind else b["_source_group"],
        "metadata": {
            "source_group_a_hidden": a["_source_group"],
            "source_group_b_hidden": b["_source_group"],
            "source_id_a": a.get("explanation_id") or a.get("generation_id"),
            "source_id_b": b.get("explanation_id") or b.get("generation_id"),
        },
    }
    return item


def build_rating_sheet_csv(
    items: Iterable[dict],
    output_path: str | Path,
    dimensions: Optional[list[str]] = None,
) -> None:
    dimensions = dimensions or DEFAULT_DIMENSIONS
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "item_id",
        "rater_id",
        "rater_skill_level",
        "rater_self_rating",
        *[f"rating_{d}" for d in dimensions],
        "overall",
        "free_text",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = {"item_id": item.get("item_id")}
            writer.writerow(row)


def ingest_ratings_csv(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ratings = {k.replace("rating_", ""): float(v) for k, v in row.items() if k.startswith("rating_") and v}
            hr = HumanRating(
                rating_id=new_id("rate"),
                item_id=row.get("item_id") or "",
                rater_id=row.get("rater_id") or None,
                rater_skill_level=row.get("rater_skill_level") or None,
                rater_self_rating=int(row.get("rater_self_rating")) if row.get("rater_self_rating") else None,
                ratings=ratings,
                overall=float(row.get("overall")) if row.get("overall") else None,
                free_text=row.get("free_text") or None,
            )
            yield asdict(hr)
