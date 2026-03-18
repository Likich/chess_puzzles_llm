from chess_reasoning.ingestion.book import iter_book_positions


def test_book_ingest(tmp_path):
    csv_content = "position_id,fen,solution_moves,section,test_id,source_ref\n1,8/8/8/8/8/8/8/K6k w - - 0 1,a1a2,Pawn Endings,Test 1,page 5\n"
    path = tmp_path / "book.csv"
    path.write_text(csv_content, encoding="utf-8")

    rows = list(iter_book_positions(path, source_name="book"))
    assert len(rows) == 1
    assert rows[0]["best_move"] == "a1a2"
    assert rows[0]["section"] == "Pawn Endings"
