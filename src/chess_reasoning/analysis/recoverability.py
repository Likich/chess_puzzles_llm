from __future__ import annotations

import re
import time
from typing import Iterable, Iterator, Optional

import chess

from chess_reasoning.evaluation.masking import mask_explanation
from chess_reasoning.generation.openai_api import create_response, extract_output_text
from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)

UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
SAN_RE = re.compile(r"\b(O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b")

DEFAULT_PROMPT = """You are given a chess position and an explanation of a move, but the move itself is masked.
Return the single best UCI move that the explanation refers to. Output only the UCI move.

FEN:
{fen}

Explanation:
{explanation}
"""

TOPK_PROMPT = """You are given a chess position and an explanation of a move, but the move itself is masked.
Return the top {k} candidate UCI moves the explanation refers to, comma-separated. Output only UCI moves.

FEN:
{fen}

Explanation:
{explanation}
"""


def _extract_moves(text: str, fen: Optional[str], top_k: int) -> list[str]:
    if not text:
        return []
    moves = [m.lower() for m in UCI_RE.findall(text)]
    if moves:
        seen = []
        for m in moves:
            if m not in seen:
                seen.append(m)
        return seen[:top_k]

    if fen:
        board = chess.Board(fen)
        tokens = SAN_RE.findall(text)
        parsed: list[str] = []
        for token in tokens:
            try:
                move = board.parse_san(token)
                uci = move.uci()
                if uci not in parsed:
                    parsed.append(uci)
            except Exception:
                continue
        return parsed[:top_k]
    return []


def predict_moves(
    fen: str,
    explanation: str,
    model: str,
    top_k: int = 1,
    temperature: float = 0.0,
    max_output_tokens: int = 64,
    sleep_s: float = 0.0,
    max_retries: int = 3,
) -> list[str]:
    prompt = DEFAULT_PROMPT.format(fen=fen, explanation=explanation)
    if top_k > 1:
        prompt = TOPK_PROMPT.format(fen=fen, explanation=explanation, k=top_k)

    start = time.time()
    response = create_response(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        sleep_s=sleep_s,
    )
    latency_ms = int((time.time() - start) * 1000)
    output_text = extract_output_text(response) or ""
    moves = _extract_moves(output_text, fen, top_k)
    logger.debug("Recoverability model latency %sms, output=%s", latency_ms, output_text)
    return moves


def evaluate_recoverability(
    rows: Iterable[dict],
    model: str,
    mask_mode: str = "strict",
    top_k: int = 1,
    temperature: float = 0.0,
    max_output_tokens: int = 64,
    sleep_s: float = 0.0,
    max_retries: int = 3,
) -> Iterator[dict]:
    for row in rows:
        fen = row.get("fen")
        if not fen:
            continue
        original_move = row.get("move_uci") or row.get("parsed_move") or row.get("chosen_move")
        if not original_move:
            continue
        reasoning_text = row.get("reasoning_text") or row.get("parsed_explanation") or row.get("clean_text") or row.get("raw_text")
        if not reasoning_text:
            continue

        masked_light = mask_explanation(reasoning_text, level="light")
        masked_strict = mask_explanation(reasoning_text, level="strict")
        masked = masked_strict if mask_mode == "strict" else masked_light

        predictions = predict_moves(
            fen=fen,
            explanation=masked,
            model=model,
            top_k=top_k,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            sleep_s=sleep_s,
            max_retries=max_retries,
        )

        predicted_move = predictions[0] if predictions else None
        recover_exact = None
        recover_top3 = None
        if predicted_move:
            recover_exact = predicted_move.lower() == original_move.lower()
        if top_k >= 3 and predictions:
            recover_top3 = original_move.lower() in [m.lower() for m in predictions[:3]]

        out = dict(row)
        out.update(
            {
                "reasoning_masked_light": masked_light,
                "reasoning_masked_strict": masked_strict,
                "mask_mode": mask_mode,
                "predicted_move": predicted_move,
                "top_k_predictions": predictions if predictions else None,
                "recoverable_exact": recover_exact,
                "recoverable_top3": recover_top3,
                "recoverable_exact_light": recover_exact if mask_mode == "light" else None,
                "recoverable_exact_strict": recover_exact if mask_mode == "strict" else None,
            }
        )
        yield out
