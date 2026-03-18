from __future__ import annotations

import re

UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


def parse_move_and_explanation(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None, None

    first = lines[0]
    match = UCI_RE.search(first)
    move = match.group(1).lower() if match else None

    if move:
        explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else None
    else:
        explanation = text.strip()
    return move, explanation
