from __future__ import annotations

import time
from typing import Iterator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_reasoning.schema import LLMGeneration, as_json
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.logging import get_logger
from chess_reasoning.generation.prompts import render_prompt
from chess_reasoning.generation.parser import parse_move_and_explanation
from chess_reasoning.evaluation.move_quality import is_legal_move, exact_best_move

logger = get_logger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return None


def _strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    return full_text.strip()


def generate_hf_rows(
    puzzles: Iterator[dict],
    prompt_template: str,
    model_id: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 256,
    prompt_condition: str = "custom",
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)

    logger.info("Loading HF model %s on %s", model_id, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=resolved_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(resolved_device)

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
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(resolved_device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                top_p=top_p if temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency_ms = int((time.time() - start) * 1000)

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_text = _strip_prompt(decoded, prompt)
        parsed_move, parsed_explanation = parse_move_and_explanation(output_text)

        legal = None
        correct = None
        if parsed_move:
            legal = is_legal_move(fen, parsed_move)
            if best_move:
                correct = exact_best_move(best_move, parsed_move)

        usage = {
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "output_tokens": int(output_ids.shape[-1] - inputs["input_ids"].shape[-1]),
        }

        row = LLMGeneration(
            generation_id=new_id("gen"),
            puzzle_id=puzzle_id,
            model_name=model_id,
            provider="hf",
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
                "usage": usage,
                "device": resolved_device,
                "dtype": str(resolved_dtype) if resolved_dtype else "auto",
            },
        )
        count += 1
        yield as_json(row)
