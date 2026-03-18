from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import chess.pgn

from chess_reasoning.schema import MoveAnnotation, as_json
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)

# NAG (Numeric Annotation Glyph) mapping to common symbols
NAG_SYMBOL_MAP: dict[int, str] = {
    1: "!",
    2: "?",
    3: "!!",
    4: "??",
    5: "!?",
    6: "?!",
}

# Priority for picking a single label when multiple NAGs exist
NAG_PRIORITY = [3, 4, 1, 2, 5, 6]


def nags_to_symbols(nags: set[int]) -> list[str]:
    symbols: list[str] = []
    for nag in sorted(nags):
        if nag in NAG_SYMBOL_MAP:
            symbols.append(NAG_SYMBOL_MAP[nag])
    return symbols


def pick_primary_nag(nags: set[int]) -> Optional[int]:
    for nag in NAG_PRIORITY:
        if nag in nags:
            return nag
    return None


def iter_pgn_move_annotations(
    input_path: str | Path,
    source_url: Optional[str] = None,
    license_name: Optional[str] = None,
) -> Iterator[dict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            headers = dict(game.headers)
            board = game.board()

            for node in game.mainline():
                move = node.move
                if move is None:
                    continue

                nags = set(node.nags)
                if not nags:
                    board.push(move)
                    continue

                primary_nag = pick_primary_nag(nags)
                annotation_symbol = NAG_SYMBOL_MAP.get(primary_nag) if primary_nag is not None else None
                all_symbols = nags_to_symbols(nags)

                annotation = MoveAnnotation(
                    annotation_id=new_id("annot"),
                    puzzle_id=None,
                    fen=board.fen(),
                    move_uci=move.uci(),
                    move_san=board.san(move),
                    annotation_symbol=annotation_symbol,
                    nag=primary_nag,
                    all_nags=sorted(nags),
                    all_symbols=all_symbols,
                    source_type="pgn_annotation",
                    source_url=source_url,
                    author=headers.get("Annotator") or headers.get("White") or headers.get("Black"),
                    license=license_name,
                    metadata={
                        "event": headers.get("Event"),
                        "site": headers.get("Site"),
                        "date": headers.get("Date"),
                        "round": headers.get("Round"),
                        "white": headers.get("White"),
                        "black": headers.get("Black"),
                        "result": headers.get("Result"),
                    },
                )

                yield as_json(annotation)
                board.push(move)
