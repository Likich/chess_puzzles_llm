from __future__ import annotations

from typing import Iterable, Iterator

from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)


def build_fen_index(puzzles: Iterable[dict]) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for p in puzzles:
        fen = p.get("fen")
        if fen and fen not in index:
            index[fen] = p
    return index


def match_explanations(
    puzzles: Iterable[dict],
    explanations: Iterable[dict],
) -> Iterator[dict]:
    fen_index = build_fen_index(puzzles)

    for exp in explanations:
        fen = exp.get("fen")
        chosen_move = exp.get("chosen_move")
        matched = False

        if fen and fen in fen_index:
            puzzle = fen_index[fen]
            exp["puzzle_id"] = puzzle.get("puzzle_id")
            exp.setdefault("metadata", {})
            exp["metadata"]["match_method"] = "fen_exact"
            exp["metadata"]["match_confidence"] = 1.0
            if chosen_move and puzzle.get("solution_moves"):
                exp["metadata"]["move_in_solution"] = chosen_move in puzzle["solution_moves"]
            matched = True

        if not matched:
            exp.setdefault("metadata", {})
            exp["metadata"]["match_method"] = "unmatched"
            exp["metadata"]["match_confidence"] = 0.0

        yield exp
