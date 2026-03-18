from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests

from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1/responses"


def extract_output_text(response_json: dict[str, Any]) -> str | None:
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
    return None


def create_response(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: int = 60,
    max_retries: int = 3,
    sleep_s: float = 0.0,
) -> dict[str, Any]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

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
