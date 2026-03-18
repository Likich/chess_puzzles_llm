from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


def as_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)


@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    solution_moves: list[str]
    best_move: str
    rating: Optional[int]
    themes: list[str]
    source: str
    split: Optional[str] = None
    game_url: Optional[str] = None
    opening_tags: Optional[list[str]] = None
    section: Optional[str] = None
    source_ref: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanExplanation:
    explanation_id: str
    puzzle_id: Optional[str]
    source_type: str
    source_url: Optional[str]
    author: Optional[str]
    license: Optional[str]
    raw_text: str
    clean_text: Optional[str]
    chosen_move: Optional[str]
    confidence: Optional[float]
    skill_level: Optional[str]
    fen: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMGeneration:
    generation_id: str
    puzzle_id: str
    model_name: str
    provider: str
    temperature: float
    prompt_condition: str
    raw_response: str
    parsed_move: Optional[str]
    parsed_explanation: Optional[str]
    legal_move: Optional[bool]
    correct_move: Optional[bool]
    api_cost: Optional[float]
    latency_ms: Optional[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoverabilityExample:
    example_id: str
    puzzle_id: str
    source_group: str
    original_move: str
    explanation_original: str
    explanation_masked: str
    predictor_model: str
    predicted_move: Optional[str]
    top_k_predictions: Optional[list[str]]
    recoverable_exact: Optional[bool]
    recoverable_top3: Optional[bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MoveAnnotation:
    annotation_id: str
    puzzle_id: Optional[str]
    fen: str
    move_uci: str
    move_san: Optional[str]
    annotation_symbol: Optional[str]
    nag: Optional[int]
    all_nags: list[int]
    all_symbols: list[str]
    source_type: str
    source_url: Optional[str]
    author: Optional[str]
    license: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RatingItem:
    item_id: str
    puzzle_id: str
    fen: str
    move_uci: str
    move_san: Optional[str]
    explanation_text: str
    source_group: Optional[str]
    source_id: Optional[str]
    prompt_condition: Optional[str]
    model_name: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanRating:
    rating_id: str
    item_id: str
    rater_id: Optional[str]
    rater_skill_level: Optional[str]
    rater_self_rating: Optional[int]
    ratings: dict[str, float]
    overall: Optional[float]
    free_text: Optional[str]
    metadata: dict[str, Any] = field(default_factory=dict)
