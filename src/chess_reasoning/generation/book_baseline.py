from __future__ import annotations

from typing import Iterable, Iterator

from chess_reasoning.schema import LLMGeneration, as_json
from chess_reasoning.utils.ids import new_id
from chess_reasoning.evaluation.move_quality import is_legal_move


def book_solution_rows(
    puzzles: Iterable[dict],
    source_label: str = "book_solution",
) -> Iterator[dict]:
    for puzzle in puzzles:
        pid = puzzle.get("puzzle_id")
        fen = puzzle.get("fen")
        best_move = puzzle.get("best_move")
        if not pid or not fen or not best_move:
            continue

        legal = is_legal_move(fen, best_move)
        row = LLMGeneration(
            generation_id=new_id("book"),
            puzzle_id=pid,
            model_name=source_label,
            provider="book",
            temperature=0.0,
            prompt_condition="book",
            raw_response="",
            parsed_move=best_move,
            parsed_explanation=None,
            legal_move=legal,
            correct_move=True,
            api_cost=None,
            latency_ms=None,
            metadata={},
        )
        yield as_json(row)
