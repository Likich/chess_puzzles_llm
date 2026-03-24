from __future__ import annotations

from typing import Iterator

import chess
import chess.engine

from chess_reasoning.utils.logging import get_logger
from chess_reasoning.evaluation.move_quality import exact_best_move, is_legal_move
from chess_reasoning.parsing.moves import parse_move_from_text

logger = get_logger(__name__)


def eval_generations_with_stockfish(
    puzzles: dict[str, dict],
    generations: Iterator[dict],
    engine_path: str,
    depth: int = 12,
) -> Iterator[dict]:
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        for gen in generations:
            pid = gen.get("puzzle_id")
            if not pid or pid not in puzzles:
                continue
            puzzle = puzzles[pid]
            fen = puzzle.get("fen")
            if not fen:
                continue

            board = chess.Board(fen)
            orig_turn = board.turn

            info_best = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = None
            if info_best.get("pv"):
                best_move = info_best["pv"][0].uci()
            best_score = None
            if info_best.get("score"):
                best_score = info_best["score"].pov(orig_turn).score(mate_score=100000)

            move_score = None
            move_uci = gen.get("parsed_move") or gen.get("chosen_move") or gen.get("move_uci")
            if not move_uci:
                text = gen.get("parsed_explanation") or gen.get("raw_response") or ""
                move_uci = parse_move_from_text(fen, text)
                if move_uci:
                    gen["parsed_move"] = move_uci
                    gen["legal_move"] = is_legal_move(fen, move_uci)
                    best_move = puzzle.get("best_move")
                    if best_move and gen.get("legal_move") is True:
                        gen["correct_move"] = exact_best_move(best_move, move_uci)
            if move_uci:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        info_move = engine.analyse(board, chess.engine.Limit(depth=depth))
                        if info_move.get("score"):
                            move_score = info_move["score"].pov(orig_turn).score(mate_score=100000)
                except ValueError:
                    move_score = None

            gen["engine_best_move"] = best_move
            gen["engine_best_score_cp"] = best_score
            gen["engine_move_score_cp"] = move_score
            if best_score is not None and move_score is not None:
                gen["engine_delta_cp"] = best_score - move_score
            else:
                gen["engine_delta_cp"] = None

            yield gen
    finally:
        engine.quit()
