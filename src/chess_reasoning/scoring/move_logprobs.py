from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import chess

from chess_reasoning.models.open_model_runner import OpenModelRunner
from chess_reasoning.utils.io import read_jsonl


SCORING_PROMPT = """Position (FEN): {fen}
Best move in UCI:"""

STYLE_PROMPTS = {
    "brief": "Position (FEN): {fen}\nBest move in UCI:",
    "calc": "Analyze step by step, then output the best move in UCI.\nPosition (FEN): {fen}\nBest move in UCI:",
    "teaching": "Think like a chess teacher and output the best move in UCI.\nPosition (FEN): {fen}\nBest move in UCI:",
}


def _load_generations(path: Optional[str]) -> dict[tuple[str, str], str]:
    if not path:
        return {}
    data = {}
    for row in read_jsonl(path):
        pid = row.get("puzzle_id")
        prompt = row.get("prompt_condition") or "scoring_only"
        move = row.get("parsed_move")
        if pid and move:
            data[(pid, prompt)] = move
    return data


def _load_stockfish(path: Optional[str]) -> dict[str, str]:
    if not path:
        return {}
    data = {}
    for row in read_jsonl(path):
        pid = row.get("puzzle_id")
        move = row.get("engine_best_move") or row.get("best_move")
        if pid and move:
            data[pid] = move
    return data


def build_prompt(fen: str, prompt_style: str, prompt_template: Optional[str] = None) -> str:
    if prompt_template:
        return prompt_template.format(fen=fen)
    if prompt_style == "scoring_only":
        return SCORING_PROMPT.format(fen=fen)
    template = STYLE_PROMPTS.get(prompt_style, SCORING_PROMPT)
    return template.format(fen=fen)


def _candidate_moves_all(board: chess.Board) -> list[str]:
    return [m.uci() for m in board.legal_moves]


def _candidate_moves_filtered(
    board: chess.Board,
    book_move: Optional[str],
    generated_move: Optional[str],
    engine_move: Optional[str],
    distractors: int = 0,
) -> list[str]:
    candidates: list[str] = []
    for move in (book_move, generated_move, engine_move):
        if move and move not in candidates:
            candidates.append(move)
    if distractors > 0:
        legal = [m.uci() for m in board.legal_moves]
        for mv in legal:
            if mv not in candidates:
                candidates.append(mv)
                if len(candidates) >= (3 + distractors):
                    break
    return candidates


def iter_scored_moves(
    puzzles: Iterable[dict],
    runner: OpenModelRunner,
    candidate_mode: str,
    prompt_style: str,
    prompt_template: Optional[str],
    generations_map: dict[tuple[str, str], str],
    stockfish_map: dict[str, str],
    distractors: int = 0,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    count = 0
    for puzzle in puzzles:
        if limit is not None and count >= limit:
            break
        pid = puzzle.get("puzzle_id")
        fen = puzzle.get("fen")
        book_move = puzzle.get("best_move")
        if not pid or not fen:
            continue
        board = chess.Board(fen)
        generated_move = generations_map.get((pid, prompt_style))
        if not generated_move and prompt_style != "scoring_only":
            generated_move = generations_map.get((pid, "scoring_only"))
        engine_move = stockfish_map.get(pid)

        if candidate_mode == "all_legal":
            candidates = _candidate_moves_all(board)
        else:
            candidates = _candidate_moves_filtered(
                board,
                book_move=book_move,
                generated_move=generated_move,
                engine_move=engine_move,
                distractors=distractors,
            )

        prompt = build_prompt(fen, prompt_style, prompt_template)

        scored = []
        for move in candidates:
            score = runner.score_completion(prompt, move)
            scored.append((move, score))

        scored_sorted = sorted(scored, key=lambda x: x[1].logprob_total, reverse=True)
        ranks = {mv: idx + 1 for idx, (mv, _) in enumerate(scored_sorted)}

        for move, score in scored:
            candidate_type = "legal"
            if book_move and move == book_move:
                candidate_type = "book"
            elif generated_move and move == generated_move:
                candidate_type = "generated"
            elif engine_move and move == engine_move:
                candidate_type = "stockfish"

            try:
                legal = chess.Move.from_uci(move) in board.legal_moves
            except ValueError:
                legal = False

            yield {
                "puzzle_id": pid,
                "prompt_style": prompt_style,
                "fen": fen,
                "book_move": book_move,
                "generated_move": generated_move,
                "candidate_move": move,
                "candidate_type": candidate_type,
                "is_legal": legal,
                "logprob_total": score.logprob_total,
                "logprob_avg_token": score.logprob_avg_token,
                "token_count": score.token_count,
                "rank_among_candidates": ranks.get(move),
            }

        count += 1


def score_moves_from_file(
    puzzles_path: str,
    model_name: str,
    candidate_mode: str,
    prompt_style: str,
    prompt_template: Optional[str],
    generations_path: Optional[str],
    stockfish_path: Optional[str],
    distractors: int = 0,
    limit: Optional[int] = None,
    device_map: str | None = "auto",
    dtype: str | None = None,
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: str | None = None,
) -> Iterator[dict]:
    runner = OpenModelRunner(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    generations_map = _load_generations(generations_path)
    stockfish_map = _load_stockfish(stockfish_path)
    puzzles = read_jsonl(puzzles_path)
    yield from iter_scored_moves(
        puzzles=puzzles,
        runner=runner,
        candidate_mode=candidate_mode,
        prompt_style=prompt_style,
        prompt_template=prompt_template,
        generations_map=generations_map,
        stockfish_map=stockfish_map,
        distractors=distractors,
        limit=limit,
    )
