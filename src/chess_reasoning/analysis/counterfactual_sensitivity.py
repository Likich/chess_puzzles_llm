from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import chess
from difflib import SequenceMatcher

from chess_reasoning.models.open_model_runner import OpenModelRunner
from chess_reasoning.scoring.move_logprobs import build_prompt, iter_scored_moves
from chess_reasoning.evaluation.masking import mask_explanation
from chess_reasoning.analysis.recoverability import predict_moves
from chess_reasoning.utils.io import read_jsonl, write_jsonl


MOVE_PROMPT = "Position (FEN): {fen}\nBest move in UCI:"
EXPLAIN_PROMPT = "Position (FEN): {fen}\nExplain the best move in one paragraph."


def _first_king_move_variant(board: chess.Board) -> Optional[chess.Board]:
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KING:
            new_board = board.copy()
            new_board.push(move)
            return new_board
    return None


def _first_pawn_move_variant(board: chess.Board) -> Optional[chess.Board]:
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            new_board = board.copy()
            new_board.push(move)
            return new_board
    return None


def generate_variants(fen: str) -> list[tuple[str, bool]]:
    board = chess.Board(fen)
    variants = [(fen, True)]
    king_variant = _first_king_move_variant(board)
    if king_variant:
        variants.append((king_variant.fen(), False))
    pawn_variant = _first_pawn_move_variant(board)
    if pawn_variant:
        variants.append((pawn_variant.fen(), False))
    return variants


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def run_counterfactuals(
    puzzles_path: str,
    model_name: str,
    prompt_style: str,
    output_path: str,
    candidate_mode: str = "all_legal",
    limit: Optional[int] = None,
    device_map: str | None = "auto",
    dtype: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: str | None = None,
    recoverability_model: Optional[str] = None,
    recoverability_max_tokens: int = 64,
    recoverability_sleep_s: float = 0.0,
) -> None:
    runner = OpenModelRunner(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    rows = []

    puzzles = list(read_jsonl(puzzles_path))
    if limit is not None:
        puzzles = puzzles[:limit]

    for puzzle in puzzles:
        pid = puzzle.get("puzzle_id")
        fen = puzzle.get("fen")
        book_move = puzzle.get("best_move")
        if not pid or not fen:
            continue

        variants = generate_variants(fen)
        original_expl = None

        for idx, (variant_fen, is_original) in enumerate(variants):
            move_prompt = build_prompt(variant_fen, prompt_style, None)
            generated_move = runner.generate_move(move_prompt, max_new_tokens=16).strip()

            explanation = runner.generate_explanation(EXPLAIN_PROMPT.format(fen=variant_fen), max_new_tokens=128).strip()
            if is_original:
                original_expl = explanation

            scored = list(
                iter_scored_moves(
                    puzzles=[{"puzzle_id": pid, "fen": variant_fen, "best_move": book_move}],
                    runner=runner,
                    candidate_mode=candidate_mode,
                    prompt_style=prompt_style,
                    prompt_template=None,
                    generations_map={},
                    stockfish_map={},
                    distractors=0,
                    limit=None,
                )
            )
            book_row = next((r for r in scored if r.get("candidate_move") == book_move), None)
            book_rank = book_row.get("rank_among_candidates") if book_row else None
            topprob_move = None
            if scored:
                best = max(scored, key=lambda r: r.get("logprob_total", float("-inf")))
                topprob_move = best.get("candidate_move")

            similarity = None
            if not is_original and original_expl is not None:
                similarity = _text_similarity(original_expl, explanation)

            predicted_move = None
            recoverability_matches_generated = None
            if recoverability_model:
                masked = mask_explanation(explanation, level="strict")
                preds = predict_moves(
                    fen=variant_fen,
                    explanation=masked,
                    model=recoverability_model,
                    top_k=1,
                    max_output_tokens=recoverability_max_tokens,
                    sleep_s=recoverability_sleep_s,
                )
                predicted_move = preds[0] if preds else None
                if predicted_move:
                    recoverability_matches_generated = predicted_move == generated_move

            rows.append(
                {
                    "puzzle_id": pid,
                    "variant_id": f"{pid}_v{idx}",
                    "is_original": is_original,
                    "fen": variant_fen,
                    "generated_move": generated_move,
                    "topprob_move": topprob_move,
                    "book_move_rank": book_rank,
                    "explanation_text": explanation,
                    "explanation_similarity_to_original": similarity,
                    "recoverability_predicted_move": predicted_move,
                    "recoverability_matches_generated": recoverability_matches_generated,
                    "engine_delta_cp": None,
                    "prompt_style": prompt_style,
                    "candidate_mode": candidate_mode,
                }
            )

    write_jsonl(output_path, rows)


def summarize_counterfactuals(rows: Iterable[dict]) -> list[dict]:
    original = [r for r in rows if r.get("is_original")]
    variants = [r for r in rows if not r.get("is_original")]

    if not original or not variants:
        return []

    change_counts = {
        "generated_move_changed": 0,
        "topprob_move_changed": 0,
        "recoverability_predicted_move_changed": 0,
        "recoverability_match_changed": 0,
    }
    total = 0
    sims = []
    rank_deltas = []

    for var in variants:
        pid = var.get("puzzle_id")
        orig = next((r for r in original if r.get("puzzle_id") == pid), None)
        if not orig:
            continue
        total += 1
        if orig.get("generated_move") != var.get("generated_move"):
            change_counts["generated_move_changed"] += 1
        if orig.get("topprob_move") != var.get("topprob_move"):
            change_counts["topprob_move_changed"] += 1
        if orig.get("recoverability_predicted_move") != var.get("recoverability_predicted_move"):
            if orig.get("recoverability_predicted_move") is not None and var.get("recoverability_predicted_move") is not None:
                change_counts["recoverability_predicted_move_changed"] += 1
        if orig.get("recoverability_matches_generated") != var.get("recoverability_matches_generated"):
            if orig.get("recoverability_matches_generated") is not None and var.get("recoverability_matches_generated") is not None:
                change_counts["recoverability_match_changed"] += 1
        if var.get("explanation_similarity_to_original") is not None:
            sims.append(var.get("explanation_similarity_to_original"))
        if orig.get("book_move_rank") is not None and var.get("book_move_rank") is not None:
            rank_deltas.append(var.get("book_move_rank") - orig.get("book_move_rank"))

    avg_sim = sum(sims) / len(sims) if sims else None
    avg_rank_delta = sum(rank_deltas) / len(rank_deltas) if rank_deltas else None

    summary = [
        {"metric": "mean_delta_book_rank", "mean_delta_original_vs_variant": avg_rank_delta},
        {"metric": "mean_explanation_similarity", "mean_delta_original_vs_variant": avg_sim},
    ]
    if total > 0:
        summary.extend(
            [
                {"metric": k, "mean_delta_original_vs_variant": v / total}
                for k, v in change_counts.items()
            ]
        )
    return summary


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
