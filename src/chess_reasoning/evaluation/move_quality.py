from __future__ import annotations

import chess


def is_legal_move(fen: str, uci: str) -> bool:
    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return False
    return move in board.legal_moves


def exact_best_move(best_move: str, predicted_move: str) -> bool:
    return best_move.strip().lower() == predicted_move.strip().lower()
