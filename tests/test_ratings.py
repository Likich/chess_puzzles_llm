from chess_reasoning.study.ratings import build_rating_items


def test_build_rating_items_single_same_move():
    puzzles = [
        {"puzzle_id": "p1", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}
    ]
    explanations = [
        {"puzzle_id": "p1", "source_group": "human", "chosen_move": "a1a2", "clean_text": "Human"},
        {"puzzle_id": "p1", "source_group": "llm", "parsed_move": "a1a2", "parsed_explanation": "LLM"},
    ]

    items = build_rating_items(puzzles, explanations, design="single", same_move=True, per_puzzle=2, seed=1)
    assert len(items) == 2
    assert all(i["puzzle_id"] == "p1" for i in items)
