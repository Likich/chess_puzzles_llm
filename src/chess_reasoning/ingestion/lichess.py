from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Iterator

import pandas as pd

from chess_reasoning.schema import Puzzle, as_json
from chess_reasoning.parsing.moves import split_solution_moves
from chess_reasoning.utils.logging import get_logger

logger = get_logger(__name__)


def _open_csv(path: Path):
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except Exception as exc:  # pragma: no cover - dependency check
            raise RuntimeError("zstandard is required to read .zst files") from exc
        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_lichess_rows(path: str | Path, chunksize: int = 100_000) -> Iterator[dict]:
    path = Path(path)
    with _open_csv(path) as f:
        for chunk in pd.read_csv(f, chunksize=chunksize):
            for _, row in chunk.iterrows():
                yield row.to_dict()


def assign_split(rng: random.Random, ratios: tuple[float, float, float]) -> str:
    r = rng.random()
    train_r, dev_r, test_r = ratios
    if r < train_r:
        return "train"
    if r < train_r + dev_r:
        return "dev"
    return "test"


def ingest_lichess(
    input_path: str | Path,
    split_strategy: str = "none",
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Iterator[dict]:
    rng = random.Random(seed)
    for row in iter_lichess_rows(input_path):
        puzzle_id = str(row.get("PuzzleId") or row.get("PuzzleId".lower()) or row.get("puzzle_id"))
        fen = row.get("FEN") or row.get("fen")
        moves = row.get("Moves") or row.get("moves")
        rating = row.get("Rating") or row.get("rating")
        themes_raw = row.get("Themes") or row.get("themes") or ""
        game_url = row.get("GameUrl") or row.get("game_url")
        opening_tags = row.get("OpeningTags") or row.get("opening_tags")

        if not puzzle_id or not fen or not moves:
            continue

        solution_moves = split_solution_moves(str(moves))
        if not solution_moves:
            continue

        themes = [t for t in str(themes_raw).split() if t]
        opening_tags_list = None
        if opening_tags and isinstance(opening_tags, str):
            opening_tags_list = [t for t in opening_tags.split() if t]

        split = None
        if split_strategy == "random":
            split = assign_split(rng, split_ratios)

        puzzle = Puzzle(
            puzzle_id=puzzle_id,
            fen=fen,
            solution_moves=solution_moves,
            best_move=solution_moves[0],
            rating=int(rating) if rating is not None and str(rating).isdigit() else None,
            themes=themes,
            source="lichess_puzzles",
            split=split,
            game_url=game_url,
            opening_tags=opening_tags_list,
        )

        yield as_json(puzzle)
