from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import torch

from chess_reasoning.models.open_model_runner import OpenModelRunner
from chess_reasoning.scoring.move_logprobs import build_prompt
from chess_reasoning.utils.io import read_jsonl, write_jsonl


def logit_lens_bookmove(
    puzzles_path: str,
    model_name: str,
    prompt_style: str,
    output_path: str,
    limit: Optional[int] = None,
    device_map: str | None = "auto",
    dtype: str | None = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    bnb_4bit_compute_dtype: str | None = None,
) -> None:
    runner = OpenModelRunner(
        model_name=model_name,
        device_map=device_map,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    tokenizer = runner.tokenizer
    model = runner.model

    rows = []
    puzzles = list(read_jsonl(puzzles_path))
    if limit is not None:
        puzzles = puzzles[:limit]

    for puzzle in puzzles:
        pid = puzzle.get("puzzle_id")
        fen = puzzle.get("fen")
        book_move = puzzle.get("best_move")
        if not pid or not fen or not book_move:
            continue

        prompt = build_prompt(fen, prompt_style, None)
        prompt_ids = tokenizer(prompt, return_tensors="pt")
        completion_ids = tokenizer(book_move, add_special_tokens=False, return_tensors="pt")
        prompt_input_ids = prompt_ids["input_ids"]
        comp_ids = completion_ids["input_ids"]
        if comp_ids.numel() == 0:
            continue

        input_ids = torch.cat([prompt_input_ids, comp_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        prompt_len = prompt_input_ids.shape[1]
        comp_len = comp_ids.shape[1]
        lm_head = model.get_output_embeddings()

        for layer_idx, hidden in enumerate(hidden_states):
            # hidden: [1, seq_len, hidden_dim]
            logprobs = []
            for i in range(comp_len):
                pos = prompt_len + i - 1
                if pos < 0:
                    continue
                logits = lm_head(hidden[0, pos])
                token_id = comp_ids[0, i]
                lp = torch.nn.functional.log_softmax(logits, dim=-1)[token_id].item()
                logprobs.append(lp)
            total = float(sum(logprobs)) if logprobs else 0.0
            rows.append(
                {
                    "puzzle_id": pid,
                    "prompt_style": prompt_style,
                    "layer_index": layer_idx,
                    "book_move": book_move,
                    "book_move_logprob_layer": total,
                    "topprob_move_layer": None,
                    "topprob_logprob_layer": None,
                }
            )

    write_jsonl(output_path, rows)
