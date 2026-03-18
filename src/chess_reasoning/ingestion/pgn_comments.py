from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import chess.pgn

from chess_reasoning.schema import HumanExplanation, as_json
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)


def iter_pgn_comments(
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
                comment = node.comment
                if comment and comment.strip():
                    explanation = HumanExplanation(
                        explanation_id=new_id("hexpl"),
                        puzzle_id=None,
                        source_type="pgn_comment",
                        source_url=source_url,
                        author=headers.get("White") or headers.get("Black"),
                        license=license_name,
                        raw_text=comment.strip(),
                        clean_text=comment.strip(),
                        chosen_move=move.uci() if move else None,
                        confidence=None,
                        skill_level=None,
                        fen=board.fen(),
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
                    yield as_json(explanation)

                if move is not None:
                    board.push(move)
