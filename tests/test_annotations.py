from chess_reasoning.ingestion.annotations import iter_pgn_move_annotations
from chess_reasoning.evaluation.annotation_scoring import extract_symbol_from_text


def test_pgn_annotation_ingest(tmp_path):
    pgn = """[Event \"Test\"]\n[Site \"Local\"]\n[Date \"2025.01.01\"]\n[Round \"1\"]\n[White \"White\"]\n[Black \"Black\"]\n[Result \"*\"]\n\n1. e4 $1 e5 $4 2. Nf3 $6 Nc6 *\n"""
    path = tmp_path / "test.pgn"
    path.write_text(pgn, encoding="utf-8")

    rows = list(iter_pgn_move_annotations(path))
    symbols = [r.get("annotation_symbol") for r in rows]
    assert "!" in symbols
    assert "??" in symbols
    assert "?!" in symbols


def test_extract_symbol_from_text():
    text = "This is a brilliant move!! It sacrifices a piece."
    assert extract_symbol_from_text(text) == "!!"
