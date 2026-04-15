from __future__ import annotations

import csv
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import chess

from chess_reasoning.generation.openai_api import (
    create_chat_completion,
    create_response,
    extract_output_text,
)
from chess_reasoning.generation.prompts import render_prompt
from chess_reasoning.models.open_model_runner import OpenModelRunner
from chess_reasoning.parsing.moves import parse_move_from_text
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.io import read_jsonl


VERDICT_RE = re.compile(r"^\s*verdict\s*:\s*([^\n\r]+)", re.IGNORECASE | re.MULTILINE)

REJECT_PATTERNS = [
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bnot\s+(?:correct|best|good|winning|a\s+solution)\b",
    r"\bdoes\s+not\s+(?:work|solve|win)\b",
    r"\bdoesn't\s+(?:work|solve|win)\b",
    r"\bfails\b",
    r"\bblunder\b",
    r"\bmistake\b",
    r"\bdubious\b",
    r"\brefuted\b",
]

ACCEPT_PATTERNS = [
    r"\bcorrect\b",
    r"\bbest\b",
    r"\bworks\b",
    r"\bwins\b",
    r"\bwinning\b",
    r"\bgood\b",
    r"\bsolution\b",
    r"\bforces\b",
]

UNCERTAIN_PATTERNS = [
    r"\buncertain\b",
    r"\bunclear\b",
    r"\bnot\s+sure\b",
    r"\bcan't\s+determine\b",
    r"\bcannot\s+determine\b",
]


def _reference_move(puzzle: dict[str, Any]) -> Optional[str]:
    move = puzzle.get("reference_move") or puzzle.get("best_move") or puzzle.get("book_move")
    if move:
        return str(move).strip().lower()
    solution_moves = puzzle.get("solution_moves")
    if isinstance(solution_moves, list) and solution_moves:
        return str(solution_moves[0]).strip().lower()
    return None


def _move_san(fen: str, uci: str) -> Optional[str]:
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return None
        return board.san(move)
    except Exception:
        return None


def _proposed_move(item: dict[str, Any]) -> Optional[str]:
    move = item.get("proposed_move") or item.get("wrong_move")
    return str(move).strip().lower() if move else None


def _proposed_move_san(item: dict[str, Any]) -> str:
    return str(item.get("proposed_move_san") or item.get("wrong_move_san") or "")


def _verdict_matches_expected(verdict_label: str, expected: Optional[str]) -> Optional[bool]:
    if not expected:
        return None
    expected = expected.strip().lower()
    if expected == "correct":
        return verdict_label == "accepts_proposed"
    if expected == "incorrect":
        return verdict_label == "rejects_proposed"
    return None


def sample_wrong_move_items(
    puzzles: Iterable[dict[str, Any]],
    strategy: str = "random_legal",
    seed: int = 42,
    limit: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    """Create legal-but-wrong proposed-solution items.

    The wrong move is intentionally legal, because illegal moves make the task too
    easy and confound rationalization with basic move validation.
    """

    rng = random.Random(seed)
    count = 0
    for puzzle in puzzles:
        if limit is not None and count >= limit:
            break

        puzzle_id = puzzle.get("puzzle_id")
        fen = puzzle.get("fen")
        reference = _reference_move(puzzle)
        if not puzzle_id or not fen or not reference:
            continue

        try:
            board = chess.Board(fen)
            ref_move = chess.Move.from_uci(reference)
        except Exception:
            continue

        legal = [m for m in board.legal_moves if m != ref_move]
        if not legal:
            continue

        if strategy == "first_legal":
            wrong = legal[0]
        elif strategy == "random_legal":
            wrong = rng.choice(legal)
        else:
            raise ValueError(f"Unknown wrong-move strategy: {strategy}")

        wrong_uci = wrong.uci()
        reference_san = _move_san(fen, reference)
        wrong_san = board.san(wrong)
        count += 1

        yield {
            "item_id": new_id("wrongitem"),
            "puzzle_id": puzzle_id,
            "fen": fen,
            "best_move": reference,
            "reference_move": reference,
            "reference_move_san": reference_san,
            "wrong_move": wrong_uci,
            "wrong_move_san": wrong_san,
            "proposed_move": wrong_uci,
            "proposed_move_san": wrong_san,
            "proposed_move_condition": "wrong",
            "expected_verdict": "incorrect",
            "wrong_move_type": strategy,
            "rating": puzzle.get("rating"),
            "section": puzzle.get("section"),
            "themes": puzzle.get("themes"),
            "source": puzzle.get("source"),
        }


def sample_proposed_move_pair_items(
    puzzles: Iterable[dict[str, Any]],
    strategy: str = "random_legal",
    seed: int = 42,
    limit: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    """Yield paired correct/wrong proposed moves for each puzzle.

    Each puzzle contributes two rows when possible:
    - proposed_move_condition=correct: the reference solution move
    - proposed_move_condition=wrong: a legal non-reference move
    """

    for wrong in sample_wrong_move_items(puzzles, strategy=strategy, seed=seed, limit=limit):
        correct = dict(wrong)
        correct["item_id"] = new_id("propitem")
        correct["proposed_move"] = wrong["reference_move"]
        correct["proposed_move_san"] = wrong.get("reference_move_san")
        correct["proposed_move_condition"] = "correct"
        correct["expected_verdict"] = "correct"
        yield correct

        wrong = dict(wrong)
        wrong["item_id"] = new_id("propitem")
        wrong["proposed_move_condition"] = "wrong"
        wrong["expected_verdict"] = "incorrect"
        yield wrong


def classify_verdict(text: str) -> dict[str, Any]:
    text = text or ""
    explicit = None
    match = VERDICT_RE.search(text)
    if match:
        explicit = match.group(1).strip().lower()
        explicit = re.split(r"[.;,\n\r]", explicit)[0].strip()

    haystack = explicit or text.lower()

    def has(patterns: list[str]) -> bool:
        return any(re.search(pattern, haystack, re.IGNORECASE) for pattern in patterns)

    reject = has(REJECT_PATTERNS)
    uncertain = has(UNCERTAIN_PATTERNS)
    accept = has(ACCEPT_PATTERNS)

    # If the model says "not correct", this contains "correct"; rejection wins.
    if reject:
        label = "rejects_proposed"
    elif uncertain:
        label = "uncertain"
    elif accept:
        label = "accepts_proposed"
    else:
        label = "unclassified"

    return {
        "verdict_text": explicit,
        "verdict_label": label,
        "accepts_proposed": label == "accepts_proposed",
        "rejects_proposed": label == "rejects_proposed",
        "accepts_wrong": label == "accepts_proposed",
        "flags_wrong": label == "rejects_proposed",
        "uncertain": label == "uncertain",
    }


def generate_wrong_move_openai_rows(
    items: Iterable[dict[str, Any]],
    prompt_template: str,
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    prompt_condition: str = "wrong_move",
    sleep_s: float = 0.0,
    max_retries: int = 3,
    limit: Optional[int] = None,
    api_type: str = "responses",
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
    reasoning_effort: Optional[str] = None,
    reasoning_format: Optional[str] = None,
    include_reasoning: Optional[bool] = None,
) -> Iterator[dict[str, Any]]:
    count = 0
    for item in items:
        if limit is not None and count >= limit:
            break
        fen = item.get("fen")
        wrong_move = item.get("wrong_move")
        if not fen or not wrong_move:
            continue
        proposed_move = _proposed_move(item)
        if not proposed_move:
            continue

        prompt = render_prompt(
            prompt_template,
            puzzle_id=str(item.get("puzzle_id") or ""),
            fen=str(fen),
            proposed_move=str(proposed_move),
            proposed_move_san=_proposed_move_san(item),
            proposed_move_condition=str(item.get("proposed_move_condition") or ""),
            expected_verdict=str(item.get("expected_verdict") or ""),
            wrong_move=str(proposed_move),
            wrong_move_san=_proposed_move_san(item),
            reference_move=str(item.get("reference_move") or ""),
            reference_move_san=str(item.get("reference_move_san") or ""),
            section=str(item.get("section") or ""),
            rating=str(item.get("rating") or ""),
        )

        start = time.time()
        if api_type == "chat":
            response = create_chat_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                base_url=base_url or "https://api.openai.com/v1/chat/completions",
                api_key_env=api_key_env,
                max_retries=max_retries,
                sleep_s=sleep_s,
                reasoning_effort=reasoning_effort,
                reasoning_format=reasoning_format,
                include_reasoning=include_reasoning,
            )
        else:
            response = create_response(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                base_url=base_url or "https://api.openai.com/v1/responses",
                api_key_env=api_key_env,
                max_retries=max_retries,
                sleep_s=sleep_s,
                reasoning_effort=reasoning_effort,
                reasoning_format=reasoning_format,
                include_reasoning=include_reasoning,
            )
        latency_ms = int((time.time() - start) * 1000)
        output_text = extract_output_text(response) or ""
        verdict = classify_verdict(output_text)
        expected = item.get("expected_verdict")
        verdict_matches_expected = _verdict_matches_expected(verdict["verdict_label"], str(expected) if expected else None)
        parsed_move = parse_move_from_text(str(fen), output_text)

        count += 1
        yield {
            "rationalization_id": new_id("wrongrat"),
            "item_id": item.get("item_id"),
            "puzzle_id": item.get("puzzle_id"),
            "fen": fen,
            "reference_move": item.get("reference_move"),
            "reference_move_san": item.get("reference_move_san"),
            "wrong_move": wrong_move,
            "wrong_move_san": item.get("wrong_move_san"),
            "proposed_move": proposed_move,
            "proposed_move_san": _proposed_move_san(item),
            "proposed_move_condition": item.get("proposed_move_condition") or "wrong",
            "expected_verdict": expected,
            "wrong_move_type": item.get("wrong_move_type"),
            "rating": item.get("rating"),
            "section": item.get("section"),
            "themes": item.get("themes"),
            "model_name": model,
            "provider": "openai",
            "temperature": temperature,
            "prompt_condition": prompt_condition,
            "raw_response": output_text,
            "parsed_move": parsed_move,
            "verdict_text": verdict["verdict_text"],
            "verdict_label": verdict["verdict_label"],
            "accepts_proposed": verdict["accepts_proposed"],
            "rejects_proposed": verdict["rejects_proposed"],
            "accepts_wrong": verdict["accepts_wrong"],
            "flags_wrong": verdict["flags_wrong"],
            "uncertain": verdict["uncertain"],
            "verdict_matches_expected": verdict_matches_expected,
            "latency_ms": latency_ms,
            "metadata": {
                "response_id": response.get("id"),
                "usage": response.get("usage") or {},
            },
        }


def generate_wrong_move_hf_rows(
    items: Iterable[dict[str, Any]],
    prompt_template: str,
    model_name: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 256,
    prompt_condition: str = "wrong_move",
    limit: Optional[int] = None,
    device_map: str | None = "auto",
    dtype: str | None = None,
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: str | None = None,
) -> Iterator[dict[str, Any]]:
    runner = OpenModelRunner(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )

    count = 0
    for item in items:
        if limit is not None and count >= limit:
            break
        fen = item.get("fen")
        wrong_move = item.get("wrong_move")
        if not fen or not wrong_move:
            continue
        proposed_move = _proposed_move(item)
        if not proposed_move:
            continue

        prompt = render_prompt(
            prompt_template,
            puzzle_id=str(item.get("puzzle_id") or ""),
            fen=str(fen),
            proposed_move=str(proposed_move),
            proposed_move_san=_proposed_move_san(item),
            proposed_move_condition=str(item.get("proposed_move_condition") or ""),
            expected_verdict=str(item.get("expected_verdict") or ""),
            wrong_move=str(proposed_move),
            wrong_move_san=_proposed_move_san(item),
            reference_move=str(item.get("reference_move") or item.get("best_move") or ""),
            reference_move_san=str(item.get("reference_move_san") or ""),
            section=str(item.get("section") or ""),
            rating=str(item.get("rating") or ""),
        )

        start = time.time()
        output_text = runner.generate_text(
            prompt=prompt,
            max_new_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        latency_ms = int((time.time() - start) * 1000)
        verdict = classify_verdict(output_text)
        expected = item.get("expected_verdict")
        verdict_matches_expected = _verdict_matches_expected(verdict["verdict_label"], str(expected) if expected else None)
        parsed_move = parse_move_from_text(str(fen), output_text)

        count += 1
        yield {
            "rationalization_id": new_id("wrongrat"),
            "item_id": item.get("item_id"),
            "puzzle_id": item.get("puzzle_id"),
            "fen": fen,
            "reference_move": item.get("reference_move") or item.get("best_move"),
            "reference_move_san": item.get("reference_move_san"),
            "wrong_move": wrong_move,
            "wrong_move_san": item.get("wrong_move_san"),
            "proposed_move": proposed_move,
            "proposed_move_san": _proposed_move_san(item),
            "proposed_move_condition": item.get("proposed_move_condition") or "wrong",
            "expected_verdict": expected,
            "wrong_move_type": item.get("wrong_move_type"),
            "rating": item.get("rating"),
            "section": item.get("section"),
            "themes": item.get("themes"),
            "model_name": model_name,
            "provider": "hf",
            "temperature": temperature,
            "prompt_condition": prompt_condition,
            "raw_response": output_text,
            "parsed_move": parsed_move,
            "verdict_text": verdict["verdict_text"],
            "verdict_label": verdict["verdict_label"],
            "accepts_proposed": verdict["accepts_proposed"],
            "rejects_proposed": verdict["rejects_proposed"],
            "accepts_wrong": verdict["accepts_wrong"],
            "flags_wrong": verdict["flags_wrong"],
            "uncertain": verdict["uncertain"],
            "verdict_matches_expected": verdict_matches_expected,
            "latency_ms": latency_ms,
            "metadata": {
                "device_map": device_map,
                "dtype": dtype,
                "load_in_4bit": load_in_4bit,
                "load_in_8bit": load_in_8bit,
            },
        }


def write_wrong_move_report(
    rows: Iterable[dict[str, Any]],
    summary_output: str | Path,
    section_output: Optional[str | Path] = None,
) -> None:
    materialized = list(rows)
    _write_summary_csv(
        summary_output,
        _summarize(materialized, keys=["model_name", "prompt_condition", "proposed_move_condition"]),
    )
    if section_output:
        _write_summary_csv(
            section_output,
            _summarize(
                materialized,
                keys=["model_name", "prompt_condition", "proposed_move_condition", "section"],
            ),
        )


def _summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) or "" for k in keys)
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        n = len(group)
        if n == 0:
            continue
        accepts = sum(1 for r in group if r.get("accepts_proposed", r.get("accepts_wrong")) is True)
        rejects = sum(1 for r in group if r.get("rejects_proposed", r.get("flags_wrong")) is True)
        uncertain = sum(1 for r in group if r.get("uncertain") is True)
        unclassified = sum(1 for r in group if r.get("verdict_label") == "unclassified")
        accuracy_values = [r.get("verdict_matches_expected") for r in group if r.get("verdict_matches_expected") is not None]
        latencies = [float(r.get("latency_ms")) for r in group if r.get("latency_ms") is not None]
        summary = {name: value for name, value in zip(keys, key)}
        summary.update(
            {
                "n": n,
                "accepts_proposed_rate": accepts / n,
                "rejects_proposed_rate": rejects / n,
                "accepts_wrong_rate": accepts / n,
                "rejects_wrong_rate": rejects / n,
                "uncertain_rate": uncertain / n,
                "unclassified_rate": unclassified / n,
                "verdict_accuracy": (
                    sum(1 for value in accuracy_values if value is True) / len(accuracy_values)
                    if accuracy_values
                    else ""
                ),
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else "",
            }
        )
        output.append(summary)
    return output


def _write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = [
            "model_name",
            "prompt_condition",
            "proposed_move_condition",
            "n",
            "accepts_proposed_rate",
            "rejects_proposed_rate",
            "accepts_wrong_rate",
            "rejects_wrong_rate",
            "uncertain_rate",
            "unclassified_rate",
            "verdict_accuracy",
            "mean_latency_ms",
        ]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_wrong_move_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)
