from __future__ import annotations

import time
from typing import Iterator, Optional

from chess_reasoning.schema import LLMGeneration, as_json
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.logging import get_logger
from chess_reasoning.generation.openai_api import create_response, create_chat_completion, extract_output_text
from chess_reasoning.generation.prompts import render_prompt
from chess_reasoning.generation.parser import parse_move_and_explanation
from chess_reasoning.evaluation.move_quality import is_legal_move, exact_best_move

logger = get_logger(__name__)


def generate_openai_rows(
    puzzles: Iterator[dict],
    prompt_template: str,
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    prompt_condition: str = "custom",
    sleep_s: float = 0.0,
    max_retries: int = 3,
    limit: Optional[int] = None,
    api_type: str = "responses",
    base_url: Optional[str] = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> Iterator[dict]:
    count = 0
    for puzzle in puzzles:
        if limit is not None and count >= limit:
            break
        puzzle_id = puzzle.get("puzzle_id")
        fen = puzzle.get("fen")
        best_move = puzzle.get("best_move")
        if not puzzle_id or not fen:
            continue

        prompt = render_prompt(prompt_template, fen=fen, best_move=best_move or "")

        start = time.time()
        if api_type == "chat":
            response = create_chat_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                sleep_s=sleep_s,
                base_url=base_url or "https://api.openai.com/v1/chat/completions",
                api_key_env=api_key_env,
            )
        else:
            response = create_response(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                sleep_s=sleep_s,
                base_url=base_url or "https://api.openai.com/v1/responses",
                api_key_env=api_key_env,
            )
        latency_ms = int((time.time() - start) * 1000)

        output_text = extract_output_text(response) or ""
        parsed_move, parsed_explanation = parse_move_and_explanation(output_text)

        legal = None
        correct = None
        if parsed_move:
            legal = is_legal_move(fen, parsed_move)
            if best_move:
                correct = exact_best_move(best_move, parsed_move)

        usage = response.get("usage") or {}

        row = LLMGeneration(
            generation_id=new_id("gen"),
            puzzle_id=puzzle_id,
            model_name=model,
            provider="openai",
            temperature=temperature,
            prompt_condition=prompt_condition,
            raw_response=output_text,
            parsed_move=parsed_move,
            parsed_explanation=parsed_explanation,
            legal_move=legal,
            correct_move=correct,
            api_cost=None,
            latency_ms=latency_ms,
            metadata={
                "response_id": response.get("id"),
                "usage": usage,
            },
        )
        count += 1
        yield as_json(row)
