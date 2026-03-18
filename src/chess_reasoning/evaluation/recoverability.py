from __future__ import annotations

from typing import Iterable, Iterator

from chess_reasoning.evaluation.masking import mask_explanation
from chess_reasoning.schema import RecoverabilityExample, as_json
from chess_reasoning.utils.ids import new_id


def build_masked_examples(
    explanations: Iterable[dict],
    source_group: str,
    predictor_model: str,
    mask_level: str = "light",
) -> Iterator[dict]:
    for exp in explanations:
        if not exp.get("puzzle_id"):
            continue
        original_move = exp.get("chosen_move") or exp.get("parsed_move") or ""
        text = exp.get("clean_text") or exp.get("parsed_explanation") or exp.get("raw_text") or ""
        if not text:
            continue
        masked = mask_explanation(text, level=mask_level)
        example = RecoverabilityExample(
            example_id=new_id("reco"),
            puzzle_id=exp["puzzle_id"],
            source_group=source_group,
            original_move=original_move,
            explanation_original=text,
            explanation_masked=masked,
            predictor_model=predictor_model,
            predicted_move=None,
            top_k_predictions=None,
            recoverable_exact=None,
            recoverable_top3=None,
        )
        yield as_json(example)
