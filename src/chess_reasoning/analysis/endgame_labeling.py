from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess


ENDGAME_THEME_MAP = {
    "pawnEndgame": "pawn_endings",
    "rookEndgame": "rook_endings",
    "queenEndgame": "queen_endings",
    "bishopEndgame": "bishop_endings",
    "knightEndgame": "knight_endings",
    "queenRookEndgame": "queen_rook_endings",
}


@dataclass
class EndgameLabel:
    section: Optional[str]
    method: Optional[str]
    details: dict


def _count_pieces(board: chess.Board) -> dict:
    counts = {
        "white": {
            "pawns": len(board.pieces(chess.PAWN, chess.WHITE)),
            "knights": len(board.pieces(chess.KNIGHT, chess.WHITE)),
            "bishops": len(board.pieces(chess.BISHOP, chess.WHITE)),
            "rooks": len(board.pieces(chess.ROOK, chess.WHITE)),
            "queens": len(board.pieces(chess.QUEEN, chess.WHITE)),
        },
        "black": {
            "pawns": len(board.pieces(chess.PAWN, chess.BLACK)),
            "knights": len(board.pieces(chess.KNIGHT, chess.BLACK)),
            "bishops": len(board.pieces(chess.BISHOP, chess.BLACK)),
            "rooks": len(board.pieces(chess.ROOK, chess.BLACK)),
            "queens": len(board.pieces(chess.QUEEN, chess.BLACK)),
        },
    }
    return counts


def _bishop_square_colors(board: chess.Board, color: bool) -> list[int]:
    squares = list(board.pieces(chess.BISHOP, color))
    colors = []
    for sq in squares:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        colors.append((file + rank) % 2)
    return colors


def _material_endgame_section(board: chess.Board) -> Optional[str]:
    counts = _count_pieces(board)
    w = counts["white"]
    b = counts["black"]

    major = w["queens"] + w["rooks"] + b["queens"] + b["rooks"]
    minors = w["bishops"] + w["knights"] + b["bishops"] + b["knights"]

    # Pawn endings: only kings + pawns
    if major == 0 and minors == 0:
        return "pawn_endings"

    # Queen endings: queens only (plus pawns/kings)
    if major > 0 and (w["rooks"] + b["rooks"] == 0) and minors == 0:
        return "queen_endings"

    # Rook endings: rooks only (plus pawns/kings)
    if (w["rooks"] + b["rooks"] > 0) and (w["queens"] + b["queens"] == 0) and minors == 0:
        return "rook_endings"

    # Mixed queen+rook (no minors)
    if (w["queens"] + b["queens"] > 0) and (w["rooks"] + b["rooks"] > 0) and minors == 0:
        return "queen_rook_endings"

    # Knight against pawns
    if major == 0 and (w["bishops"] + b["bishops"] == 0):
        if (w["knights"] > 0 and b["knights"] == 0) or (b["knights"] > 0 and w["knights"] == 0):
            return "knight_vs_pawns"

    # Bishop against pawns
    if major == 0 and (w["knights"] + b["knights"] == 0):
        if (w["bishops"] > 0 and b["bishops"] == 0) or (b["bishops"] > 0 and w["bishops"] == 0):
            return "bishop_vs_pawns"

    # Knight endings (knights on both sides, no bishops/major pieces)
    if major == 0 and (w["bishops"] + b["bishops"] == 0) and (w["knights"] + b["knights"] > 0):
        return "knight_endings"

    # Bishop vs Knight endings (no major pieces)
    if major == 0:
        if (w["bishops"] > 0 and b["knights"] > 0 and b["bishops"] == 0 and w["knights"] == 0) or (
            b["bishops"] > 0 and w["knights"] > 0 and w["bishops"] == 0 and b["knights"] == 0
        ):
            return "bishop_vs_knight_endings"

    # Bishop endings (bishops only, no knights/major pieces)
    if major == 0 and (w["knights"] + b["knights"] == 0) and (w["bishops"] + b["bishops"] > 0):
        # Check for same/opposite colored bishops if one bishop each
        if w["bishops"] == 1 and b["bishops"] == 1:
            w_colors = _bishop_square_colors(board, chess.WHITE)
            b_colors = _bishop_square_colors(board, chess.BLACK)
            if w_colors and b_colors:
                if w_colors[0] == b_colors[0]:
                    return "same_colored_bishop_endings"
                return "opposite_colored_bishop_endings"
        return "bishop_endings"

    return None


def label_endgame_section(
    fen: Optional[str],
    themes: list[str] | None,
    method: str = "hybrid",
    require_endgame_tag: bool = True,
) -> EndgameLabel:
    themes = themes or []
    theme_section = None
    for t in themes:
        if t in ENDGAME_THEME_MAP:
            theme_section = ENDGAME_THEME_MAP[t]
            break

    has_endgame_theme = "endgame" in themes or theme_section is not None
    if require_endgame_tag and not has_endgame_theme:
        return EndgameLabel(section=None, method=None, details={"reason": "no_endgame_theme"})

    if method in {"theme", "hybrid"} and theme_section:
        return EndgameLabel(section=theme_section, method="theme", details={"theme": theme_section})

    if fen:
        board = chess.Board(fen)
        section = _material_endgame_section(board)
        if section:
            return EndgameLabel(section=section, method="material", details={})

    return EndgameLabel(section=None, method=None, details={"reason": "unclassified"})


def apply_endgame_labels(
    row: dict,
    method: str = "hybrid",
    require_endgame_tag: bool = True,
) -> dict:
    fen = row.get("fen")
    themes = row.get("themes") or []
    label = label_endgame_section(
        fen=fen,
        themes=themes,
        method=method,
        require_endgame_tag=require_endgame_tag,
    )
    out = dict(row)
    if label.section:
        out["section"] = label.section
    if label.method:
        meta = dict(out.get("metadata") or {})
        meta["endgame_label_method"] = label.method
        meta["endgame_label_details"] = label.details
        out["metadata"] = meta
    return out
