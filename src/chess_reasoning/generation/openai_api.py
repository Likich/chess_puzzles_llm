from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests

from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1/responses"
DEFAULT_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def extract_output_text(response_json: dict[str, Any]) -> str | None:
    # OpenAI Responses API
    if isinstance(response_json.get("output_text"), str):
        return response_json.get("output_text")

    texts: list[str] = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if text:
                    texts.append(text)
    if texts:
        return "\n".join(texts)

    # OpenAI-compatible Chat Completions
    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
            reasoning = msg.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                return reasoning
    return None


def create_response(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = "OPENAI_API_KEY",
    timeout_s: int = 60,
    max_retries: int = 3,
    sleep_s: float = 0.0,
    reasoning_effort: Optional[str] = None,
    reasoning_format: Optional[str] = None,
    include_reasoning: Optional[bool] = None,
) -> dict[str, Any]:
    api_key = api_key or os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if reasoning_format is not None and include_reasoning is not None:
        raise RuntimeError("reasoning_format and include_reasoning are mutually exclusive")
    if reasoning_format is not None:
        payload["reasoning_format"] = reasoning_format
    if include_reasoning is not None:
        payload["include_reasoning"] = include_reasoning

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        if sleep_s > 0:
            time.sleep(sleep_s)
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code == 200:
                return resp.json()
            msg = f"OpenAI API error {resp.status_code}: {resp.text}"
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                time.sleep(backoff)
                continue
            raise RuntimeError(msg)
        except Exception as exc:  # pragma: no cover - network failures
            last_err = exc
            if attempt < max_retries:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                time.sleep(backoff)
                continue
            raise

    if last_err:
        raise last_err
    raise RuntimeError("OpenAI API call failed")


def create_chat_completion(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    base_url: str = DEFAULT_CHAT_URL,
    api_key_env: str = "OPENAI_API_KEY",
    timeout_s: int = 60,
    max_retries: int = 3,
    sleep_s: float = 0.0,
    reasoning_effort: Optional[str] = None,
    reasoning_format: Optional[str] = None,
    include_reasoning: Optional[bool] = None,
) -> dict[str, Any]:
    api_key = api_key or os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if reasoning_format is not None and include_reasoning is not None:
        raise RuntimeError("reasoning_format and include_reasoning are mutually exclusive")
    if reasoning_format is not None:
        payload["reasoning_format"] = reasoning_format
    if include_reasoning is not None:
        payload["include_reasoning"] = include_reasoning

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        if sleep_s > 0:
            time.sleep(sleep_s)
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code == 200:
                return resp.json()
            msg = f"Chat API error {resp.status_code}: {resp.text}"
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                time.sleep(backoff)
                continue
            raise RuntimeError(msg)
        except Exception as exc:  # pragma: no cover - network failures
            last_err = exc
            if attempt < max_retries:
                backoff = min(8.0, 0.5 * (2 ** attempt))
                time.sleep(backoff)
                continue
            raise

    if last_err:
        raise last_err
    raise RuntimeError("Chat API call failed")
