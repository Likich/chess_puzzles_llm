from __future__ import annotations

import re

UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", re.IGNORECASE)
SAN_RE = re.compile(
    r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?)\b"
)
BEST_MOVE_PHRASE_RE = re.compile(r"\b(best move is|the best move is|play|plays)\b", re.IGNORECASE)

STRICT_PHRASES = [
    r"queen sacrifice on [a-h][1-8]",
    r"sacrifice on [a-h][1-8]",
    r"checkmate on [a-h][1-8]",
    r"promot(?:e|ion) on [a-h][1-8]",
    r"(moving|move|play|plays|push|advance|bring|place)\s+(the\s+)?(king|queen|rook|bishop|knight|pawn)\s+to\s+[a-h][1-8]",
    r"(king|queen|rook|bishop|knight|pawn)\s+to\s+[a-h][1-8]",
    r"(king|queen|rook|bishop|knight|pawn)\s+on\s+[a-h][1-8]",
    r"[a-h]-pawn\s+to\s+[a-h][1-8]",
]


def mask_explanation(text: str, level: str = "light") -> str:
    masked = text
    masked = UCI_RE.sub("[MOVE]", masked)
    masked = SAN_RE.sub("[MOVE]", masked)
    masked = BEST_MOVE_PHRASE_RE.sub("[PHRASE]", masked)

    if level == "strict":
        for pat in STRICT_PHRASES:
            masked = re.sub(pat, "[MASKED_PHRASE]", masked, flags=re.IGNORECASE)

    # normalize whitespace
    masked = " ".join(masked.split())
    return masked
