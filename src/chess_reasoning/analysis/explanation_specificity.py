from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from chess_reasoning.utils.io import read_jsonl, read_yaml, write_jsonl

DEFAULT_CONFIG = {
    "conditional_markers": [
        "if",
        "then",
        "after",
        "because",
        "otherwise",
        "so that",
        "therefore",
        "hence",
    ],
    "tactical_keywords": [
        "fork",
        "pin",
        "skewer",
        "opposition",
        "zugzwang",
        "promotion",
        "passed pawn",
        "tempo",
        "triangulation",
        "blockade",
        "mating net",
        "pawn race",
    ],
    "generic_phrases": [
        "improve position",
        "gain advantage",
        "support the pawn",
        "control key squares",
        "centralize",
        "restrict the king",
        "maintain control",
    ],
    "weights": {
        "square_mentions": 2.0,
        "move_mentions": 1.5,
        "conditional_markers": 1.5,
        "tactical_keywords": 1.0,
        "line_depth": 1.5,
        "generic_phrase_count": 1.0,
    },
    "normalize_by_length": True,
}

SQUARE_RE = re.compile(r"\b[a-h][1-8]\b", re.IGNORECASE)
MOVE_TOKEN_RE = re.compile(
    r"\b([a-h][1-8][a-h][1-8][qrbn]?|O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b",
    re.IGNORECASE,
)


def load_specificity_config(path: Optional[str]) -> dict:
    if path:
        data = read_yaml(path)
        if data:
            merged = DEFAULT_CONFIG.copy()
            merged.update(data)
            if "weights" in data:
                weights = DEFAULT_CONFIG["weights"].copy()
                weights.update(data.get("weights") or {})
                merged["weights"] = weights
            return merged
    return DEFAULT_CONFIG.copy()


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _phrase_pattern(phrase: str) -> re.Pattern:
    tokens = phrase.split()
    if len(tokens) > 1:
        pattern = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
    else:
        pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.compile(pattern, re.IGNORECASE)


def _count_phrases(text: str, phrases: Iterable[str]) -> int:
    count = 0
    for phrase in phrases:
        if not phrase:
            continue
        pattern = _phrase_pattern(phrase)
        count += len(pattern.findall(text))
    return count


def estimate_line_depth(text: str) -> int:
    segments = re.split(r"[.;\n]", text)
    max_depth = 0
    for seg in segments:
        moves = MOVE_TOKEN_RE.findall(seg)
        if len(moves) > max_depth:
            max_depth = len(moves)
    return max_depth


def compute_specificity_features(text: str, config: dict) -> dict:
    text = _normalize_text(text or "")
    square_mentions = len(SQUARE_RE.findall(text))
    unique_squares = len({s.lower() for s in SQUARE_RE.findall(text)})
    move_mentions = len(MOVE_TOKEN_RE.findall(text))
    conditional_markers = _count_phrases(text, config.get("conditional_markers", []))
    tactical_keyword_count = _count_phrases(text, config.get("tactical_keywords", []))
    generic_phrase_count = _count_phrases(text, config.get("generic_phrases", []))
    line_depth = estimate_line_depth(text)
    word_count = len(text.split()) if text else 0

    weights = config.get("weights", {})
    specificity_score = (
        weights.get("square_mentions", 0.0) * square_mentions
        + weights.get("move_mentions", 0.0) * move_mentions
        + weights.get("conditional_markers", 0.0) * conditional_markers
        + weights.get("tactical_keywords", 0.0) * tactical_keyword_count
        + weights.get("line_depth", 0.0) * line_depth
        - weights.get("generic_phrase_count", 0.0) * generic_phrase_count
    )

    specificity_score_norm = None
    if config.get("normalize_by_length", True):
        specificity_score_norm = specificity_score / max(1, word_count)

    return {
        "square_mentions": square_mentions,
        "unique_squares": unique_squares,
        "move_mentions": move_mentions,
        "conditional_markers": conditional_markers,
        "tactical_keyword_count": tactical_keyword_count,
        "generic_phrase_count": generic_phrase_count,
        "line_depth": line_depth,
        "word_count": word_count,
        "specificity_score": specificity_score,
        "specificity_score_norm": specificity_score_norm,
    }


def extract_explanation(row: dict) -> str:
    for key in (
        "reasoning_text",
        "explanation",
        "parsed_explanation",
        "clean_text",
        "raw_text",
        "llm_explanation",
        "transcript_clean",
        "transcript_raw",
    ):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def iter_rows(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    else:
        yield from read_jsonl(path)


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        rows = list(rows)
        if not rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            return
        fieldnames = sorted({k for row in rows for k in row.keys()})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        write_jsonl(path, rows)


def add_specificity_features(
    rows: Iterable[dict],
    config: dict,
) -> Iterator[dict]:
    for row in rows:
        text = extract_explanation(row)
        features = compute_specificity_features(text, config)
        out = dict(row)
        out.update(features)
        out["reasoning_text"] = text if text else row.get("reasoning_text")
        yield out
