from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

ALLOWED_SYMBOLS = ["!!", "??", "!?", "?!", "!", "?"]

SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9])(!\!|\?\?|!\?|\?!|!|\?)(?![A-Za-z0-9])")


def normalize_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    sym = symbol.strip()
    if sym in ALLOWED_SYMBOLS:
        return sym
    return None


def extract_symbol_from_text(text: str) -> str | None:
    if not text:
        return None
    matches = SYMBOL_RE.findall(text)
    if not matches:
        return None
    # Prefer the first occurrence, but normalize to the allowed set
    for m in matches:
        sym = normalize_symbol(m)
        if sym:
            return sym
    return None


def score_annotations(
    rows: Iterable[dict],
    gold_field: str = "annotation_symbol",
    pred_field: str = "predicted_symbol",
    pred_text_field: str | None = None,
) -> dict:
    total = 0
    correct = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    per_label = Counter()
    per_label_correct = Counter()

    for row in rows:
        gold = normalize_symbol(row.get(gold_field))
        pred = normalize_symbol(row.get(pred_field))
        if pred is None and pred_text_field:
            pred = extract_symbol_from_text(row.get(pred_text_field, "") or "")

        if gold is None or pred is None:
            continue

        total += 1
        per_label[gold] += 1
        if pred == gold:
            correct += 1
            per_label_correct[gold] += 1
        confusion[gold][pred] += 1

    accuracy = correct / total if total else 0.0
    per_label_acc = {k: (per_label_correct[k] / per_label[k]) for k in per_label}

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_label_accuracy": per_label_acc,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }
