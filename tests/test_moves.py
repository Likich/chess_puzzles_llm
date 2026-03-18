from chess_reasoning.parsing.moves import split_solution_moves, normalize_uci


def test_split_solution_moves():
    moves = split_solution_moves("e2e4 e7e5 g1f3")
    assert moves == ["e2e4", "e7e5", "g1f3"]


def test_normalize_uci():
    assert normalize_uci("E2E4 ") == "e2e4"
