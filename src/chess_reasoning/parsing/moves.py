from __future__ import annotations

import chess


def normalize_uci(uci: str) -> str:
    return uci.strip().lower().replace(" ", "")


def is_legal_uci(fen: str, uci: str) -> bool:
    board = chess.Board(fen)
    move = chess.Move.from_uci(normalize_uci(uci))
    return move in board.legal_moves


def uci_to_san(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    move = chess.Move.from_uci(normalize_uci(uci))
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {uci} for FEN {fen}")
    return board.san(move)


def parse_san(fen: str, san: str) -> str:
    board = chess.Board(fen)
    move = board.parse_san(san)
    return move.uci()


def split_solution_moves(moves_str: str) -> list[str]:
    moves = [normalize_uci(m) for m in moves_str.strip().split() if m.strip()]
    return moves
