from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ScoredCompletion:
    logprob_total: float
    logprob_avg_token: float
    token_count: int
    tokens: list[int]


class OpenModelRunner:
    def __init__(
        self,
        model_name: str,
        device_map: str | None = "auto",
        dtype: str | None = None,
        trust_remote_code: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: str | None = None,
    ) -> None:
        dtype_map = {
            None: None,
            "auto": None,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, None)
        if load_in_4bit and load_in_8bit:
            raise ValueError("Only one of load_in_4bit/load_in_8bit can be True")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        quantization_config = None
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig

            if bnb_4bit_compute_dtype is None:
                compute_dtype = torch_dtype if torch_dtype in {torch.float16, torch.bfloat16} else torch.float16
            else:
                compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.float16)

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=trust_remote_code,
            quantization_config=quantization_config,
        )
        self.model.eval()

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def score_completion(self, prompt: str, completion: str) -> ScoredCompletion:
        if completion is None:
            return ScoredCompletion(0.0, 0.0, 0, [])

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")
        completion_ids = self.tokenizer(completion, add_special_tokens=False, return_tensors="pt")

        prompt_input_ids = prompt_ids["input_ids"]
        comp_ids = completion_ids["input_ids"]
        if comp_ids.numel() == 0:
            return ScoredCompletion(0.0, 0.0, 0, [])

        input_ids = torch.cat([prompt_input_ids, comp_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        device = self._device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        prompt_len = prompt_input_ids.shape[1]
        comp_len = comp_ids.shape[1]

        logprobs = []
        log_softmax = torch.nn.functional.log_softmax
        for i in range(comp_len):
            pos = prompt_len + i - 1
            if pos < 0:
                continue
            token_id = comp_ids[0, i]
            lp = log_softmax(logits[0, pos], dim=-1)[token_id].item()
            logprobs.append(lp)

        total = float(sum(logprobs)) if logprobs else 0.0
        avg = total / len(logprobs) if logprobs else 0.0
        return ScoredCompletion(total, avg, len(logprobs), comp_ids[0].tolist())

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = self._device()
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        do_sample = temperature > 0.0
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_move(self, prompt: str, max_new_tokens: int = 16) -> str:
        return self.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)

    def generate_explanation(self, prompt: str, max_new_tokens: int = 128) -> str:
        return self.generate_text(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
