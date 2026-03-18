from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator, Optional

from chess_reasoning.schema import HumanExplanation, as_json
from chess_reasoning.utils.ids import new_id
from chess_reasoning.utils.logging import get_logger
from chess_reasoning.parsing.fen_tools import san_line_to_uci

logger = get_logger(__name__)

FILLER_RE = re.compile(r"\b(um+|uh+|like|you know|i mean)\b", re.IGNORECASE)


def _clean_text(text: str, strip_fillers: bool) -> str:
    cleaned = " ".join((text or "").split())
    if strip_fillers:
        cleaned = FILLER_RE.sub("", cleaned)
        cleaned = " ".join(cleaned.split())
    return cleaned


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    val = value.strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def ingest_human_transcripts(
    input_path: str | Path,
    puzzles: dict[str, dict],
    strip_fillers: bool = False,
    license_name: Optional[str] = None,
) -> Iterator[dict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = (row.get("puzzle_id") or "").strip() or None
            puzzle = puzzles.get(puzzle_id) if puzzle_id else None
            fen = (row.get("fen") or (puzzle.get("fen") if puzzle else None) or "").strip() or None
            transcript_raw = row.get("transcript_raw") or row.get("transcript") or ""
            transcript_clean = _clean_text(transcript_raw, strip_fillers=strip_fillers)

            move_uci = (row.get("move_uci") or "").strip() or None
            move_san = (row.get("move_san") or "").strip() or None
            if not move_uci and move_san and fen:
                try:
                    uci_moves = san_line_to_uci(fen, move_san)
                    move_uci = uci_moves[0] if uci_moves else None
                except Exception:
                    move_uci = None

            participant_id = (row.get("participant_id") or "").strip() or None
            response_id = (row.get("response_id") or "").strip() or None

            confidence = _parse_float(row.get("confidence"))
            time_seconds = _parse_float(row.get("time_seconds"))

            explanation = HumanExplanation(
                explanation_id=new_id("htran"),
                puzzle_id=puzzle_id,
                source_type="human_transcript",
                source_url=None,
                author=participant_id,
                license=license_name,
                raw_text=transcript_raw,
                clean_text=transcript_clean,
                chosen_move=move_uci,
                confidence=confidence,
                skill_level=row.get("skill_level") or None,
                fen=fen,
                metadata={
                    "response_id": response_id,
                    "participant_id": participant_id,
                    "move_san": move_san,
                    "move_uci": move_uci,
                    "time_seconds": time_seconds,
                    "audio_file": row.get("audio_file") or None,
                    "notes": row.get("notes") or None,
                    "book_move": puzzle.get("best_move") if puzzle else None,
                    "section": puzzle.get("section") if puzzle else None,
                    "source": puzzle.get("source") if puzzle else None,
                },
            )
            yield as_json(explanation)
