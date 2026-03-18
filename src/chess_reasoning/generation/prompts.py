from __future__ import annotations

from pathlib import Path

from chess_reasoning.utils.io import read_text


def load_prompt(path: str | Path) -> str:
    return read_text(path)


def render_prompt(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)
