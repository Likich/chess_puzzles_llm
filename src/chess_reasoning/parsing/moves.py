from __future__ import annotations

import re

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


UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
UCI_DASH_RE = re.compile(r"\b([a-h][1-8])[-x]([a-h][1-8])([qrbn]?)\b", re.IGNORECASE)
SAN_TOKEN_RE = re.compile(
    r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b",
    re.IGNORECASE,
)


def parse_move_from_text(fen: str, text: str) -> str | None:
    if not text:
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    board = chess.Board(fen)
    candidate_lines = lines[:4]

    def _skip_line(line: str) -> bool:
        lower = line.lower()
        return "fen" in lower or "/" in line

    for line in candidate_lines:
        if _skip_line(line):
            continue
        match = UCI_RE.search(line)
        if match:
            return normalize_uci(match.group(1))
        dash_match = UCI_DASH_RE.search(line)
        if dash_match:
            return normalize_uci("".join(dash_match.groups()))

    for line in candidate_lines:
        if _skip_line(line):
            continue
        cleaned = re.sub(
            r"^(move|best move|solution|the best move is|best)\s*[:\-]?\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        for token in SAN_TOKEN_RE.findall(cleaned):
            try:
                move = board.parse_san(token)
            except ValueError:
                continue
            return move.uci()

    match = UCI_RE.search(text)
    if match:
        return normalize_uci(match.group(1))
    dash_match = UCI_DASH_RE.search(text)
    if dash_match:
        return normalize_uci("".join(dash_match.groups()))

    return None


def split_solution_moves(moves_str: str) -> list[str]:
    moves = [normalize_uci(m) for m in moves_str.strip().split() if m.strip()]
    return moves
