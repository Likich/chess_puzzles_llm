from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

from chess_reasoning.ingestion.lichess import ingest_lichess
from chess_reasoning.ingestion.pgn_comments import iter_pgn_comments
from chess_reasoning.ingestion.stackexchange import ingest_stackexchange_html
from chess_reasoning.ingestion.annotations import iter_pgn_move_annotations
from chess_reasoning.alignment.matcher import match_explanations
from chess_reasoning.evaluation.annotation_scoring import score_annotations
from chess_reasoning.utils.io import write_jsonl, read_jsonl, write_text, read_text
from chess_reasoning.utils.logging import get_logger
from chess_reasoning.study.ratings import (
    build_rating_items,
    build_rating_sheet_csv,
    ingest_ratings_csv,
    DEFAULT_DIMENSIONS,
)
from chess_reasoning.analysis.ratings_analysis import analyze_ratings
from chess_reasoning.analysis.section_analysis import summarize_by_section
from chess_reasoning.analysis.endgame_labeling import apply_endgame_labels
from chess_reasoning.analysis.explanation_specificity import (
    load_specificity_config,
    add_specificity_features,
    iter_rows as iter_specificity_rows,
    write_rows as write_specificity_rows,
)
from chess_reasoning.analysis.reasoning_comparison import build_reasoning_table, write_reasoning_reports
from chess_reasoning.analysis.recoverability import evaluate_recoverability
from chess_reasoning.scoring.move_logprobs import score_moves_from_file
from chess_reasoning.analysis.move_rank_analysis import aggregate_from_file, move_rank_report
from chess_reasoning.analysis.explanation_alignment import build_alignment_table, summarize_alignment, write_csv as write_alignment_csv
from chess_reasoning.analysis.counterfactual_sensitivity import run_counterfactuals, summarize_counterfactuals, write_csv as write_counterfactual_csv
from chess_reasoning.analysis.logit_lens import logit_lens_bookmove
from chess_reasoning.analysis.prob_recoverability import compute_prob_recoverability
from chess_reasoning.ingestion.sample import iter_filtered_puzzles, reservoir_sample
from chess_reasoning.utils.io import append_jsonl
from chess_reasoning.generation.llm_generate import generate_openai_rows
from chess_reasoning.generation.hf_generate import generate_hf_rows
from chess_reasoning.generation.prompts import load_prompt
from chess_reasoning.generation.book_baseline import book_solution_rows
from chess_reasoning.ingestion.book import iter_book_positions, iter_book_positions_sheet
from chess_reasoning.ingestion.human_transcripts import ingest_human_transcripts
from chess_reasoning.evaluation.stockfish_eval import eval_generations_with_stockfish
from chess_reasoning.parsing.fen_tools import validate_fen, san_line_to_uci

logger = get_logger(__name__)


def parse_split_ratios(value: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("split-ratios must be 'train,dev,test'")
    ratios = tuple(float(p) for p in parts)  # type: ignore
    if sum(ratios) <= 0:
        raise ValueError("split-ratios must sum to > 0")
    return ratios  # type: ignore


def cmd_ingest_lichess(args: argparse.Namespace) -> None:
    ratios = parse_split_ratios(args.split_ratios)
    rows = ingest_lichess(
        input_path=args.input,
        split_strategy=args.split_strategy,
        split_ratios=ratios,
        seed=args.seed,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote puzzles to %s", args.output)


def cmd_ingest_pgn_comments(args: argparse.Namespace) -> None:
    rows = iter_pgn_comments(
        input_path=args.input,
        source_url=args.source_url,
        license_name=args.license,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote explanations to %s", args.output)


def cmd_ingest_stackexchange(args: argparse.Namespace) -> None:
    selectors = args.selectors.split(",") if args.selectors else None
    rows = ingest_stackexchange_html(
        input_path=args.input,
        source_url=args.source_url,
        author=args.author,
        license_name=args.license,
        selectors=selectors,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote explanations to %s", args.output)


def cmd_ingest_pgn_annotations(args: argparse.Namespace) -> None:
    rows = iter_pgn_move_annotations(
        input_path=args.input,
        source_url=args.source_url,
        license_name=args.license,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote move annotations to %s", args.output)


def cmd_match_explanations(args: argparse.Namespace) -> None:
    puzzles = list(read_jsonl(args.puzzles))
    explanations = read_jsonl(args.explanations)
    rows = match_explanations(puzzles, explanations)
    write_jsonl(args.output, rows)
    logger.info("Wrote matched explanations to %s", args.output)


def cmd_score_annotations(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    metrics = score_annotations(
        rows,
        gold_field=args.gold_field,
        pred_field=args.pred_field,
        pred_text_field=args.pred_text_field,
    )
    write_text(args.output, json.dumps(metrics, ensure_ascii=True, indent=2) + "\n")
    logger.info("Wrote annotation scoring to %s", args.output)


def cmd_build_rating_items(args: argparse.Namespace) -> None:
    puzzles = list(read_jsonl(args.puzzles))
    explanations = list(read_jsonl(args.explanations))
    dimensions = args.dimensions.split(",") if args.dimensions else DEFAULT_DIMENSIONS

    items = build_rating_items(
        puzzles=puzzles,
        explanations=explanations,
        design=args.design,
        same_move=not args.allow_different_move,
        per_puzzle=args.per_puzzle,
        max_puzzles=args.max_puzzles,
        seed=args.seed,
        blind=not args.unblind,
    )

    write_jsonl(args.output, items)
    logger.info("Wrote rating items to %s", args.output)

    if args.sheet_output:
        build_rating_sheet_csv(items, args.sheet_output, dimensions=dimensions)
        logger.info("Wrote rating sheet template to %s", args.sheet_output)


def cmd_ingest_ratings(args: argparse.Namespace) -> None:
    rows = ingest_ratings_csv(args.input)
    write_jsonl(args.output, rows)
    logger.info("Wrote ratings to %s", args.output)


def cmd_analyze_ratings(args: argparse.Namespace) -> None:
    items = {r["item_id"]: r for r in read_jsonl(args.items) if r.get("item_id")}
    ratings = list(read_jsonl(args.ratings))
    metrics = analyze_ratings(ratings, items)
    write_text(args.output, json.dumps(metrics, ensure_ascii=True, indent=2) + "\n")
    logger.info("Wrote ratings analysis to %s", args.output)


def cmd_sample_puzzles(args: argparse.Namespace) -> None:
    rows = iter_filtered_puzzles(
        input_path=args.input,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        split=args.split,
    )
    if args.max_samples is None:
        write_jsonl(args.output, rows)
    else:
        sampled = reservoir_sample(rows, k=args.max_samples, seed=args.seed)
        write_jsonl(args.output, sampled)
    logger.info("Wrote sampled puzzles to %s", args.output)


def cmd_generate_openai(args: argparse.Namespace) -> None:
    prompt_template = load_prompt(args.prompt)
    prompt_condition = args.prompt_condition or Path(args.prompt).stem
    puzzles = iter_filtered_puzzles(
        input_path=args.puzzles,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        split=args.split,
    )
    rows = generate_openai_rows(
        puzzles=puzzles,
        prompt_template=prompt_template,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        prompt_condition=prompt_condition,
        sleep_s=args.sleep_s,
        max_retries=args.max_retries,
        limit=args.limit,
        api_type=args.api_type,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort,
        reasoning_format=args.reasoning_format,
        include_reasoning=args.include_reasoning,
    )
    if args.append:
        append_jsonl(args.output, rows)
    else:
        write_jsonl(args.output, rows)
    logger.info("Wrote generations to %s", args.output)


def cmd_generate_hf(args: argparse.Namespace) -> None:
    prompt_template = load_prompt(args.prompt)
    prompt_condition = args.prompt_condition or Path(args.prompt).stem
    puzzles = iter_filtered_puzzles(
        input_path=args.puzzles,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        split=args.split,
    )
    rows = generate_hf_rows(
        puzzles=puzzles,
        prompt_template=prompt_template,
        model_id=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        prompt_condition=prompt_condition,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
    )
    if args.append:
        append_jsonl(args.output, rows)
    else:
        write_jsonl(args.output, rows)
    logger.info("Wrote HF generations to %s", args.output)


def cmd_book_generations(args: argparse.Namespace) -> None:
    puzzles = list(read_jsonl(args.puzzles))
    rows = book_solution_rows(puzzles, source_label=args.source_label)
    write_jsonl(args.output, rows)
    logger.info("Wrote book generations to %s", args.output)


def cmd_ingest_book(args: argparse.Namespace) -> None:
    rows = iter_book_positions(
        input_path=args.input,
        source_name=args.source,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote book puzzles to %s", args.output)


def cmd_ingest_book_sheet(args: argparse.Namespace) -> None:
    errors: list[dict] = []
    rows = iter_book_positions_sheet(
        input_path=args.input,
        source_name=args.source,
        line=args.line,
        errors=errors,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote book puzzles to %s", args.output)
    if args.errors:
        write_jsonl(args.errors, errors)
        logger.info("Wrote book ingestion errors to %s", args.errors)


def cmd_stockfish_eval(args: argparse.Namespace) -> None:
    puzzles = {p["puzzle_id"]: p for p in read_jsonl(args.puzzles) if p.get("puzzle_id")}
    generations = read_jsonl(args.generations)
    rows = eval_generations_with_stockfish(
        puzzles=puzzles,
        generations=generations,
        engine_path=args.engine_path,
        depth=args.depth,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote Stockfish-evaluated generations to %s", args.output)


def cmd_analyze_sections(args: argparse.Namespace) -> None:
    puzzles = {p["puzzle_id"]: p for p in read_jsonl(args.puzzles) if p.get("puzzle_id")}
    generations = list(read_jsonl(args.generations))
    metrics = summarize_by_section(puzzles, generations)
    write_text(args.output, json.dumps(metrics, ensure_ascii=True, indent=2) + "\n")
    logger.info("Wrote section analysis to %s", args.output)


def cmd_explanation_specificity(args: argparse.Namespace) -> None:
    config = load_specificity_config(args.config)
    rows = iter_specificity_rows(args.input)
    enriched = add_specificity_features(rows, config)
    write_specificity_rows(args.output, enriched)
    logger.info("Wrote specificity features to %s", args.output)


def cmd_ingest_human_transcripts(args: argparse.Namespace) -> None:
    puzzles = {p["puzzle_id"]: p for p in read_jsonl(args.puzzles) if p.get("puzzle_id")}
    rows = ingest_human_transcripts(
        input_path=args.input,
        puzzles=puzzles,
        strip_fillers=args.strip_fillers,
        license_name=args.license,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote human transcripts to %s", args.output)


def cmd_build_reasoning_table(args: argparse.Namespace) -> None:
    puzzles = {}
    if args.puzzles:
        puzzles = {p["puzzle_id"]: p for p in read_jsonl(args.puzzles) if p.get("puzzle_id")}
    llm_rows = list(read_jsonl(args.llm)) if args.llm else []
    human_rows = list(read_jsonl(args.human)) if args.human else []
    book_rows = list(read_jsonl(args.book)) if args.book else []
    rows = build_reasoning_table(puzzles, llm_rows=llm_rows, human_rows=human_rows, book_rows=book_rows)
    write_jsonl(args.output, rows)
    logger.info("Wrote reasoning comparison table to %s", args.output)


def cmd_recoverability(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    output_rows = evaluate_recoverability(
        rows=rows,
        model=args.model,
        mask_mode=args.mask_mode,
        top_k=args.top_k,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        sleep_s=args.sleep_s,
        max_retries=args.max_retries,
    )
    write_jsonl(args.output, output_rows)
    logger.info("Wrote recoverability results to %s", args.output)


def cmd_reasoning_report(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.input))
    write_reasoning_reports(rows, args.output_dir)
    logger.info("Wrote reasoning reports to %s", args.output_dir)


def cmd_score_moves(args: argparse.Namespace) -> None:
    prompt_template = None
    if args.prompt:
        prompt_template = read_text(args.prompt)
    rows = score_moves_from_file(
        puzzles_path=args.input,
        model_name=args.model,
        candidate_mode=args.candidate_mode,
        prompt_style=args.prompt_style,
        prompt_template=prompt_template,
        generations_path=args.generations,
        stockfish_path=args.stockfish,
        distractors=args.distractors,
        limit=args.limit,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )
    write_jsonl(args.output, rows)
    logger.info("Wrote move logprobs to %s", args.output)


def cmd_aggregate_move_ranks(args: argparse.Namespace) -> None:
    aggregated = aggregate_from_file(args.input, args.output)
    logger.info("Wrote move rank table to %s", args.output)


def cmd_move_rank_report(args: argparse.Namespace) -> None:
    if args.input.endswith(".csv"):
        with Path(args.input).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        rows = list(read_jsonl(args.input))
    move_rank_report(rows, args.output_dir)
    logger.info("Wrote move rank report to %s", args.output_dir)


def cmd_explanation_alignment(args: argparse.Namespace) -> None:
    rows = build_alignment_table(
        generations_path=args.generations,
        logprob_path=args.logprobs,
        recoverability_path=args.recoverability,
    )
    write_alignment_csv(args.output, rows)
    summary = summarize_alignment(rows)
    if args.summary_output:
        write_alignment_csv(args.summary_output, summary)
    logger.info("Wrote explanation alignment table to %s", args.output)


def cmd_counterfactual_sensitivity(args: argparse.Namespace) -> None:
    run_counterfactuals(
        puzzles_path=args.input,
        model_name=args.model,
        prompt_style=args.prompt_style,
        output_path=args.output,
        candidate_mode=args.candidate_mode,
        limit=args.limit,
        device_map=args.device_map,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        recoverability_model=args.recoverability_model,
        recoverability_max_tokens=args.recoverability_max_tokens,
        recoverability_sleep_s=args.recoverability_sleep_s,
    )
    rows = list(read_jsonl(args.output))
    summary = summarize_counterfactuals(rows)
    if args.summary_output:
        write_counterfactual_csv(args.summary_output, summary)
    logger.info("Wrote counterfactual results to %s", args.output)


def cmd_logit_lens_bookmove(args: argparse.Namespace) -> None:
    logit_lens_bookmove(
        puzzles_path=args.input,
        model_name=args.model,
        prompt_style=args.prompt_style,
        output_path=args.output,
        limit=args.limit,
        device_map=args.device_map,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )
    logger.info("Wrote logit lens results to %s", args.output)


def cmd_prob_recoverability(args: argparse.Namespace) -> None:
    compute_prob_recoverability(
        rank_path=args.rank,
        recoverability_path=args.recoverability,
        merged_output=args.merged_output,
        rank_bucket_output=args.rank_bucket_output,
        margin_bucket_output=args.margin_bucket_output,
    )
    logger.info("Wrote probability-recoverability tables to %s", args.output_dir)


def cmd_label_endgames(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    labeled = (
        apply_endgame_labels(
            row,
            method=args.method,
            require_endgame_tag=not args.include_non_endgames,
        )
        for row in rows
    )
    write_jsonl(args.output, labeled)
    logger.info("Wrote endgame-labeled puzzles to %s", args.output)


def cmd_validate_fen(args: argparse.Namespace) -> None:
    ok, err = validate_fen(args.fen)
    if ok:
        if args.output:
            write_text(args.output, "OK\n")
        else:
            print("OK")
    else:
        msg = f"ERROR: {err}"
        if args.output:
            write_text(args.output, msg + "\n")
        else:
            print(msg)


def cmd_san_to_uci(args: argparse.Namespace) -> None:
    moves = san_line_to_uci(args.fen, args.san)
    output = " ".join(moves) + "\n"
    if args.output:
        write_text(args.output, output)
    else:
        print(output, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chess-reasoning")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest-lichess", help="Ingest Lichess puzzle dataset")
    p_ingest.add_argument("--input", required=True, help="Path to Lichess puzzle CSV or .zst")
    p_ingest.add_argument("--output", required=True, help="Output JSONL path")
    p_ingest.add_argument("--split-strategy", default="none", choices=["none", "random"])
    p_ingest.add_argument("--split-ratios", default="0.8,0.1,0.1")
    p_ingest.add_argument("--seed", type=int, default=42)
    p_ingest.set_defaults(func=cmd_ingest_lichess)

    p_pgn = sub.add_parser("ingest-pgn-comments", help="Ingest PGN comments")
    p_pgn.add_argument("--input", required=True, help="Path to PGN file")
    p_pgn.add_argument("--output", required=True, help="Output JSONL path")
    p_pgn.add_argument("--source-url", default=None)
    p_pgn.add_argument("--license", default=None)
    p_pgn.set_defaults(func=cmd_ingest_pgn_comments)

    p_se = sub.add_parser("ingest-stackexchange", help="Ingest StackExchange HTML file")
    p_se.add_argument("--input", required=True, help="Path to HTML file")
    p_se.add_argument("--output", required=True, help="Output JSONL path")
    p_se.add_argument("--source-url", default=None)
    p_se.add_argument("--author", default=None)
    p_se.add_argument("--license", default=None)
    p_se.add_argument(
        "--selectors",
        default=None,
        help="Comma-separated CSS selectors (default: .post-text,article)",
    )
    p_se.set_defaults(func=cmd_ingest_stackexchange)

    p_match = sub.add_parser("match-explanations", help="Match explanations to puzzles")
    p_match.add_argument("--puzzles", required=True, help="Puzzles JSONL")
    p_match.add_argument("--explanations", required=True, help="Explanations JSONL")
    p_match.add_argument("--output", required=True, help="Output JSONL")
    p_match.set_defaults(func=cmd_match_explanations)

    p_ann = sub.add_parser("ingest-pgn-annotations", help="Ingest PGN move annotations (NAGs)")
    p_ann.add_argument("--input", required=True, help="Path to PGN file")
    p_ann.add_argument("--output", required=True, help="Output JSONL path")
    p_ann.add_argument("--source-url", default=None)
    p_ann.add_argument("--license", default=None)
    p_ann.set_defaults(func=cmd_ingest_pgn_annotations)

    p_score = sub.add_parser("score-annotations", help="Score annotation symbol predictions")
    p_score.add_argument("--input", required=True, help="JSONL with gold + predictions")
    p_score.add_argument("--output", required=True, help="Output JSON file")
    p_score.add_argument("--gold-field", default="annotation_symbol")
    p_score.add_argument("--pred-field", default="predicted_symbol")
    p_score.add_argument("--pred-text-field", default=None)
    p_score.set_defaults(func=cmd_score_annotations)

    p_rate = sub.add_parser("build-rating-items", help="Build blind human-rating items")
    p_rate.add_argument("--puzzles", required=True, help="Puzzles JSONL")
    p_rate.add_argument("--explanations", required=True, help="Explanations JSONL (human + llm)")
    p_rate.add_argument("--output", required=True, help="Output JSONL items")
    p_rate.add_argument("--sheet-output", default=None, help="CSV template for human ratings")
    p_rate.add_argument("--dimensions", default=None, help="Comma-separated rating dimensions")
    p_rate.add_argument("--design", default="single", choices=["single", "pairwise"])
    p_rate.add_argument("--per-puzzle", type=int, default=2)
    p_rate.add_argument("--max-puzzles", type=int, default=None)
    p_rate.add_argument("--seed", type=int, default=42)
    p_rate.add_argument("--allow-different-move", action="store_true")
    p_rate.add_argument("--unblind", action="store_true")
    p_rate.set_defaults(func=cmd_build_rating_items)

    p_ingest_r = sub.add_parser("ingest-ratings", help="Ingest human ratings CSV")
    p_ingest_r.add_argument("--input", required=True, help="Ratings CSV")
    p_ingest_r.add_argument("--output", required=True, help="Output JSONL")
    p_ingest_r.set_defaults(func=cmd_ingest_ratings)

    p_an_r = sub.add_parser("analyze-ratings", help="Analyze human ratings")
    p_an_r.add_argument("--items", required=True, help="Rating items JSONL")
    p_an_r.add_argument("--ratings", required=True, help="Ratings JSONL")
    p_an_r.add_argument("--output", required=True, help="Output JSON report")
    p_an_r.set_defaults(func=cmd_analyze_ratings)

    p_sample = sub.add_parser("sample-puzzles", help="Sample puzzles with rating filters")
    p_sample.add_argument("--input", required=True, help="Puzzles JSONL")
    p_sample.add_argument("--output", required=True, help="Output JSONL")
    p_sample.add_argument("--min-rating", type=int, default=None)
    p_sample.add_argument("--max-rating", type=int, default=None)
    p_sample.add_argument("--split", default=None, choices=["train", "dev", "test"])
    p_sample.add_argument("--max-samples", type=int, default=None)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.set_defaults(func=cmd_sample_puzzles)

    p_gen = sub.add_parser("generate-openai", help="Generate LLM moves + explanations via OpenAI API")
    p_gen.add_argument("--puzzles", required=True, help="Puzzles JSONL")
    p_gen.add_argument("--prompt", required=True, help="Prompt template file")
    p_gen.add_argument("--output", required=True, help="Output JSONL")
    p_gen.add_argument("--model", required=True, help="Model name, e.g., gpt-4.1-mini")
    p_gen.add_argument("--temperature", type=float, default=0.0)
    p_gen.add_argument("--max-output-tokens", type=int, default=256)
    p_gen.add_argument("--prompt-condition", default=None, help="Label for prompt condition")
    p_gen.add_argument("--api-type", default="responses", choices=["responses", "chat"])
    p_gen.add_argument("--base-url", default=None, help="Override API base URL")
    p_gen.add_argument("--api-key-env", default="OPENAI_API_KEY", help="API key env var name")
    p_gen.add_argument("--reasoning-effort", default=None, help="Reasoning effort (model-specific)")
    p_gen.add_argument("--reasoning-format", default=None, help="Reasoning format: hidden, raw, parsed")
    p_gen.add_argument("--include-reasoning", action="store_true", default=None, help="Include reasoning field")
    p_gen.add_argument("--sleep-s", type=float, default=0.0)
    p_gen.add_argument("--max-retries", type=int, default=3)
    p_gen.add_argument("--limit", type=int, default=None)
    p_gen.add_argument("--append", action="store_true")
    p_gen.add_argument("--min-rating", type=int, default=None)
    p_gen.add_argument("--max-rating", type=int, default=None)
    p_gen.add_argument("--split", default=None, choices=["train", "dev", "test"])
    p_gen.set_defaults(func=cmd_generate_openai)

    p_hf = sub.add_parser("generate-hf", help="Generate LLM moves + explanations via Hugging Face model")
    p_hf.add_argument("--puzzles", required=True, help="Puzzles JSONL")
    p_hf.add_argument("--prompt", required=True, help="Prompt template file")
    p_hf.add_argument("--output", required=True, help="Output JSONL")
    p_hf.add_argument("--model", required=True, help="HF model id, e.g., Likich/qwen2p5-7b-chess-sft")
    p_hf.add_argument("--temperature", type=float, default=0.0)
    p_hf.add_argument("--top-p", type=float, default=1.0)
    p_hf.add_argument("--max-output-tokens", type=int, default=256)
    p_hf.add_argument("--prompt-condition", default=None, help="Label for prompt condition")
    p_hf.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p_hf.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p_hf.add_argument("--trust-remote-code", action="store_true")
    p_hf.add_argument("--limit", type=int, default=None)
    p_hf.add_argument("--append", action="store_true")
    p_hf.add_argument("--min-rating", type=int, default=None)
    p_hf.add_argument("--max-rating", type=int, default=None)
    p_hf.add_argument("--split", default=None, choices=["train", "dev", "test"])
    p_hf.set_defaults(func=cmd_generate_hf)

    p_book_gen = sub.add_parser("make-book-generations", help="Create generations from book best moves")
    p_book_gen.add_argument("--puzzles", required=True, help="Puzzles JSONL")
    p_book_gen.add_argument("--output", required=True, help="Output JSONL")
    p_book_gen.add_argument("--source-label", default="book_solution")
    p_book_gen.set_defaults(func=cmd_book_generations)

    p_book = sub.add_parser("ingest-book", help="Ingest book positions from CSV")
    p_book.add_argument("--input", required=True, help="Input CSV with FEN + solutions")
    p_book.add_argument("--output", required=True, help="Output JSONL")
    p_book.add_argument("--source", required=True, help="Source name label")
    p_book.set_defaults(func=cmd_ingest_book)

    p_book_sheet = sub.add_parser("ingest-book-sheet", help="Ingest extended book CSV (with SAN/metadata)")
    p_book_sheet.add_argument("--input", required=True, help="Input CSV from spreadsheet")
    p_book_sheet.add_argument("--output", required=True, help="Output JSONL")
    p_book_sheet.add_argument("--source", required=True, help="Source name label")
    p_book_sheet.add_argument("--line", default="full", choices=["full", "main"])
    p_book_sheet.add_argument("--errors", default=None, help="Optional JSONL for parse errors")
    p_book_sheet.set_defaults(func=cmd_ingest_book_sheet)

    p_sf = sub.add_parser("stockfish-eval", help="Evaluate generations with Stockfish")
    p_sf.add_argument("--puzzles", required=True, help="Puzzles JSONL")
    p_sf.add_argument("--generations", required=True, help="Generations JSONL")
    p_sf.add_argument("--output", required=True, help="Output JSONL")
    p_sf.add_argument("--engine-path", required=True, help="Path to Stockfish binary")
    p_sf.add_argument("--depth", type=int, default=12)
    p_sf.set_defaults(func=cmd_stockfish_eval)

    p_sec = sub.add_parser("analyze-sections", help="Analyze results by section")
    p_sec.add_argument("--puzzles", required=True, help="Puzzles JSONL")
    p_sec.add_argument("--generations", required=True, help="Generations JSONL")
    p_sec.add_argument("--output", required=True, help="Output JSON report")
    p_sec.set_defaults(func=cmd_analyze_sections)

    p_spec = sub.add_parser("explanation-specificity", help="Score explanation specificity features")
    p_spec.add_argument("--input", required=True, help="Input JSONL or CSV")
    p_spec.add_argument("--output", required=True, help="Output JSONL or CSV")
    p_spec.add_argument("--config", default=None, help="YAML config for keyword lists and weights")
    p_spec.set_defaults(func=cmd_explanation_specificity)

    p_ht = sub.add_parser("ingest-human-transcripts", help="Ingest human think-aloud transcripts")
    p_ht.add_argument("--input", required=True, help="Input CSV with transcripts")
    p_ht.add_argument("--puzzles", required=True, help="Puzzles JSONL (for FEN lookup)")
    p_ht.add_argument("--output", required=True, help="Output JSONL")
    p_ht.add_argument("--license", default=None, help="License or consent label")
    p_ht.add_argument("--strip-fillers", action="store_true")
    p_ht.set_defaults(func=cmd_ingest_human_transcripts)

    p_rt = sub.add_parser("build-reasoning-table", help="Combine LLM, human, and book reasoning")
    p_rt.add_argument("--puzzles", default=None, help="Puzzles JSONL for FEN and book moves")
    p_rt.add_argument("--llm", default=None, help="LLM generations JSONL")
    p_rt.add_argument("--human", default=None, help="Human transcripts JSONL")
    p_rt.add_argument("--book", default=None, help="Book generations JSONL")
    p_rt.add_argument("--output", required=True, help="Output JSONL")
    p_rt.set_defaults(func=cmd_build_reasoning_table)

    p_rec = sub.add_parser("recoverability", help="Evaluate move recoverability from explanations")
    p_rec.add_argument("--input", required=True, help="Input JSONL with explanations")
    p_rec.add_argument("--output", required=True, help="Output JSONL")
    p_rec.add_argument("--mask-mode", default="strict", choices=["light", "strict"])
    p_rec.add_argument("--model", required=True, help="Model name for predictor")
    p_rec.add_argument("--top-k", type=int, default=1)
    p_rec.add_argument("--temperature", type=float, default=0.0)
    p_rec.add_argument("--max-output-tokens", type=int, default=64)
    p_rec.add_argument("--sleep-s", type=float, default=0.0)
    p_rec.add_argument("--max-retries", type=int, default=3)
    p_rec.set_defaults(func=cmd_recoverability)

    p_rep = sub.add_parser("reasoning-report", help="Generate reasoning summary tables")
    p_rep.add_argument("--input", required=True, help="Recoverability JSONL")
    p_rep.add_argument("--output-dir", required=True, help="Output directory for CSVs")
    p_rep.set_defaults(func=cmd_reasoning_report)

    p_score = sub.add_parser("score-moves", help="Score candidate moves with token logprobs")
    p_score.add_argument("--input", required=True, help="Puzzles JSONL")
    p_score.add_argument("--model", required=True, help="Open-weights model id")
    p_score.add_argument("--candidate-mode", default="all_legal", choices=["all_legal", "filtered"])
    p_score.add_argument("--prompt-style", default="scoring_only", choices=["scoring_only", "brief", "calc", "teaching"])
    p_score.add_argument("--prompt", default=None, help="Optional prompt template file")
    p_score.add_argument("--generations", default=None, help="LLM generations JSONL for generated move")
    p_score.add_argument("--stockfish", default=None, help="Stockfish-evaluated JSONL for engine best move")
    p_score.add_argument("--distractors", type=int, default=0)
    p_score.add_argument("--limit", type=int, default=None)
    p_score.add_argument("--device-map", default="auto")
    p_score.add_argument("--dtype", default=None, choices=[None, "auto", "float16", "bfloat16", "float32"])
    p_score.add_argument("--trust-remote-code", action="store_true")
    p_score.add_argument("--load-in-4bit", action="store_true")
    p_score.add_argument("--load-in-8bit", action="store_true")
    p_score.add_argument("--bnb-4bit-compute-dtype", default=None, choices=[None, "float16", "bfloat16", "float32"])
    p_score.add_argument("--output", required=True, help="Output JSONL")
    p_score.set_defaults(func=cmd_score_moves)

    p_agg = sub.add_parser("aggregate-move-ranks", help="Aggregate per-puzzle move ranks")
    p_agg.add_argument("--input", required=True, help="Move logprobs JSONL")
    p_agg.add_argument("--output", required=True, help="Output CSV")
    p_agg.set_defaults(func=cmd_aggregate_move_ranks)

    p_mrr = sub.add_parser("move-rank-report", help="Summarize move-rank metrics")
    p_mrr.add_argument("--input", required=True, help="Move-rank JSONL or aggregated JSONL")
    p_mrr.add_argument("--output-dir", required=True, help="Output directory")
    p_mrr.set_defaults(func=cmd_move_rank_report)

    p_align = sub.add_parser("explanation-alignment", help="Compute explanation alignment vs move ranks")
    p_align.add_argument("--generations", required=True, help="LLM generations JSONL")
    p_align.add_argument("--logprobs", required=True, help="Move logprobs JSONL")
    p_align.add_argument("--recoverability", default=None, help="Recoverability JSONL")
    p_align.add_argument("--output", required=True, help="Output CSV")
    p_align.add_argument("--summary-output", default=None, help="Summary CSV output")
    p_align.set_defaults(func=cmd_explanation_alignment)

    p_cf = sub.add_parser("counterfactual-sensitivity", help="Run counterfactual sensitivity analysis")
    p_cf.add_argument("--input", required=True, help="Puzzles JSONL")
    p_cf.add_argument("--model", required=True, help="Open-weights model id")
    p_cf.add_argument("--prompt-style", default="scoring_only", choices=["scoring_only", "brief", "calc", "teaching"])
    p_cf.add_argument("--candidate-mode", default="all_legal", choices=["all_legal", "filtered"])
    p_cf.add_argument("--limit", type=int, default=None)
    p_cf.add_argument("--device-map", default="auto")
    p_cf.add_argument("--dtype", default=None, choices=[None, "auto", "float16", "bfloat16", "float32"])
    p_cf.add_argument("--load-in-4bit", action="store_true")
    p_cf.add_argument("--load-in-8bit", action="store_true")
    p_cf.add_argument("--bnb-4bit-compute-dtype", default=None, choices=[None, "float16", "bfloat16", "float32"])
    p_cf.add_argument("--recoverability-model", default=None, help="Model name for recoverability decoder")
    p_cf.add_argument("--recoverability-max-tokens", type=int, default=64)
    p_cf.add_argument("--recoverability-sleep-s", type=float, default=0.0)
    p_cf.add_argument("--output", required=True, help="Output JSONL")
    p_cf.add_argument("--summary-output", default=None, help="Summary CSV output")
    p_cf.set_defaults(func=cmd_counterfactual_sensitivity)

    p_ll = sub.add_parser("logit-lens-bookmove", help="Compute logit-lens book move probabilities")
    p_ll.add_argument("--input", required=True, help="Puzzles JSONL")
    p_ll.add_argument("--model", required=True, help="Open-weights model id")
    p_ll.add_argument("--prompt-style", default="scoring_only", choices=["scoring_only", "brief", "calc", "teaching"])
    p_ll.add_argument("--limit", type=int, default=None)
    p_ll.add_argument("--device-map", default="auto")
    p_ll.add_argument("--dtype", default=None, choices=[None, "auto", "float16", "bfloat16", "float32"])
    p_ll.add_argument("--load-in-4bit", action="store_true")
    p_ll.add_argument("--load-in-8bit", action="store_true")
    p_ll.add_argument("--bnb-4bit-compute-dtype", default=None, choices=[None, "float16", "bfloat16", "float32"])
    p_ll.add_argument("--output", required=True, help="Output JSONL")
    p_ll.set_defaults(func=cmd_logit_lens_bookmove)

    p_pr = sub.add_parser("prob-recoverability", help="Link move probabilities to recoverability")
    p_pr.add_argument("--rank", required=True, help="Per-puzzle move rank CSV")
    p_pr.add_argument("--recoverability", required=True, help="Recoverability JSONL")
    p_pr.add_argument("--merged-output", required=True, help="Merged per-puzzle CSV output")
    p_pr.add_argument("--rank-bucket-output", required=True, help="Rank bucket summary CSV")
    p_pr.add_argument("--margin-bucket-output", required=True, help="Margin bucket summary CSV")
    p_pr.add_argument("--output-dir", default="outputs/tables", help="Output directory label for logs")
    p_pr.set_defaults(func=cmd_prob_recoverability)

    p_end = sub.add_parser("label-endgames", help="Label Lichess puzzles by endgame section")
    p_end.add_argument("--input", required=True, help="Puzzles JSONL")
    p_end.add_argument("--output", required=True, help="Output JSONL")
    p_end.add_argument("--method", default="hybrid", choices=["theme", "material", "hybrid"])
    p_end.add_argument(
        "--include-non-endgames",
        action="store_true",
        help="Label even if no endgame theme is present",
    )
    p_end.set_defaults(func=cmd_label_endgames)

    p_fen = sub.add_parser("validate-fen", help="Validate a FEN string")
    p_fen.add_argument("--fen", required=True, help="FEN string")
    p_fen.add_argument("--output", default=None, help="Optional output file")
    p_fen.set_defaults(func=cmd_validate_fen)

    p_san = sub.add_parser("san-to-uci", help="Convert SAN line to UCI using a FEN")
    p_san.add_argument("--fen", required=True, help="FEN string")
    p_san.add_argument("--san", required=True, help="SAN moves, e.g. '1. Kf2 Kf7'")
    p_san.add_argument("--output", default=None, help="Optional output file")
    p_san.set_defaults(func=cmd_san_to_uci)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
