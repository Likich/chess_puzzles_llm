from __future__ import annotations

import chess


def validate_fen(fen: str) -> tuple[bool, str | None]:
    try:
        chess.Board(fen)
        return True, None
    except Exception as exc:
        return False, str(exc)


def san_line_to_uci(fen: str, san_line: str) -> list[str]:
    board = chess.Board(fen)
    moves: list[str] = []
    tokens = [t.strip() for t in san_line.replace("\n", " ").split() if t.strip()]

    for token in tokens:
        # normalize tokens like "1.Kf2" or "1...Kf7"
        if "." in token and token[0].isdigit():
            token = token.split("...")[-1]
            token = token.split(".")[-1]
            token = token.strip()
            if not token:
                continue
        # skip bare move numbers like "1." or "23..."
        if token.endswith(".") or token.replace(".", "").isdigit():
            continue
        # strip annotations like !? or ??
        token = token.replace("!!", "").replace("!?", "").replace("?!", "").replace("??", "").replace("!", "").replace("?", "")
        move = board.parse_san(token)
        moves.append(move.uci())
        board.push(move)
    return moves
