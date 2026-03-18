from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from chess_reasoning.schema import Puzzle, as_json
from chess_reasoning.parsing.moves import split_solution_moves
from chess_reasoning.parsing.fen_tools import validate_fen, san_line_to_uci
from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)


def iter_book_positions(
    input_path: str | Path,
    source_name: str,
) -> Iterator[dict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            position_id = row.get("position_id") or row.get("id")
            puzzle_id = row.get("puzzle_id") or (f"book_{position_id}" if position_id else None)
            fen = row.get("fen")
            moves = row.get("solution_moves") or row.get("solution")
            section = row.get("section") or row.get("chapter")
            source_ref = row.get("source_ref") or row.get("page")
            test_id = row.get("test_id")

            if not puzzle_id or not fen or not moves:
                continue

            solution_moves = split_solution_moves(str(moves))
            if not solution_moves:
                continue

            puzzle = Puzzle(
                puzzle_id=puzzle_id,
                fen=fen,
                solution_moves=solution_moves,
                best_move=solution_moves[0],
                rating=None,
                themes=[],
                source=source_name,
                split=None,
                game_url=None,
                opening_tags=None,
                section=section,
                source_ref=source_ref,
                metadata={
                    "test_id": test_id,
                },
            )
            yield as_json(puzzle)


def iter_book_positions_sheet(
    input_path: str | Path,
    source_name: str,
    line: str = "full",
    errors: list[dict] | None = None,
) -> Iterator[dict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = row.get("puzzle_id") or row.get("position_id")
            fen = row.get("fen")
            section = row.get("section") or row.get("chapter")
            page = row.get("page")
            position_index = row.get("position_index")
            source_ref = row.get("source_ref") or (f"page {page} diagram {position_index}" if page and position_index else None)

            if not puzzle_id or not fen:
                if errors is not None:
                    errors.append({"puzzle_id": puzzle_id, "error": "missing puzzle_id or fen"})
                continue

            ok, fen_err = validate_fen(fen)
            if not ok:
                if errors is not None:
                    errors.append({"puzzle_id": puzzle_id, "error": f"invalid fen: {fen_err}"})
                continue

            if line == "main":
                uci_line = row.get("solution_uci_main") or ""
                san_line = row.get("solution_san_main") or ""
            else:
                uci_line = row.get("solution_uci_full") or ""
                san_line = row.get("solution_san_full") or ""

            if uci_line.strip().upper() == "SAN_TO_UCI_LATER":
                uci_line = ""

            if not uci_line.strip() and san_line.strip():
                try:
                    uci_moves = san_line_to_uci(fen, san_line)
                    uci_line = " ".join(uci_moves)
                except Exception as exc:
                    if errors is not None:
                        errors.append({"puzzle_id": puzzle_id, "error": f"san parse failed: {exc}"})
                    continue

            if not uci_line.strip():
                if errors is not None:
                    errors.append({"puzzle_id": puzzle_id, "error": "missing solution line"})
                continue

            solution_moves = split_solution_moves(uci_line)
            if not solution_moves:
                if errors is not None:
                    errors.append({"puzzle_id": puzzle_id, "error": "solution moves empty after parsing"})
                continue

            puzzle = Puzzle(
                puzzle_id=puzzle_id,
                fen=fen,
                solution_moves=solution_moves,
                best_move=solution_moves[0],
                rating=None,
                themes=[],
                source=source_name,
                split=None,
                game_url=None,
                opening_tags=None,
                section=section,
                source_ref=source_ref,
                metadata={
                    "book_title": row.get("book_title"),
                    "chapter": row.get("chapter"),
                    "page": page,
                    "position_index": position_index,
                    "test_number": row.get("test_number"),
                    "author_label": row.get("author_label"),
                    "year_label": row.get("year_label"),
                    "objective": row.get("objective"),
                    "book_result": row.get("book_result"),
                    "marks": row.get("marks"),
                    "has_variation": row.get("has_variation"),
                    "variation_san": row.get("variation_san"),
                    "notes": row.get("notes"),
                    "source_pdf": row.get("source_pdf"),
                    "page_solution": row.get("page_solution"),
                    "entry_status": row.get("entry_status"),
                    "entered_by": row.get("entered_by"),
                    "verified_fen": row.get("verified_fen"),
                    "verified_solution": row.get("verified_solution"),
                },
            )
            yield as_json(puzzle)
