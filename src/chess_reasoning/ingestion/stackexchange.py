from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from bs4 import BeautifulSoup

from chess_reasoning.schema import HumanExplanation, as_json
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_SELECTORS = [
    ".post-text",  # Stack Exchange
    "article",     # generic
]


def extract_text(html: str, selectors: Optional[list[str]] = None) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    selectors = selectors or DEFAULT_SELECTORS
    texts: list[str] = []
    for sel in selectors:
        for node in soup.select(sel):
            text = " ".join(node.get_text(" ").split())
            if text:
                texts.append(text)
    if not texts:
        body = soup.get_text(" ")
        text = " ".join(body.split())
        if text:
            texts.append(text)
    return texts


def ingest_stackexchange_html(
    input_path: str | Path,
    source_url: Optional[str],
    author: Optional[str],
    license_name: Optional[str],
    selectors: Optional[list[str]] = None,
) -> Iterator[dict]:
    html = Path(input_path).read_text(encoding="utf-8")
    texts = extract_text(html, selectors=selectors)
    for text in texts:
        explanation = HumanExplanation(
            explanation_id=new_id("hexpl"),
            puzzle_id=None,
            source_type="stackexchange",
            source_url=source_url,
            author=author,
            license=license_name,
            raw_text=text,
            clean_text=text,
            chosen_move=None,
            confidence=None,
            skill_level=None,
            fen=None,
            metadata={},
        )
        yield as_json(explanation)
