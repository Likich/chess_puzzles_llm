from chess_reasoning.analysis.wrong_move_rationalization import (
    classify_verdict,
    sample_proposed_move_pair_items,
    sample_wrong_move_items,
    write_wrong_move_report,
)


def test_sample_wrong_move_items_returns_legal_non_reference_move():
    puzzles = [
        {
            "puzzle_id": "p1",
            "fen": "k7/8/8/8/8/8/5K2/8 w - - 0 1",
            "best_move": "f2e2",
            "rating": 1000,
            "section": "king_endings",
        }
    ]

    rows = list(sample_wrong_move_items(puzzles, strategy="first_legal"))

    assert len(rows) == 1
    assert rows[0]["wrong_move"] != "f2e2"
    assert rows[0]["reference_move"] == "f2e2"
    assert rows[0]["proposed_move"] == rows[0]["wrong_move"]
    assert rows[0]["expected_verdict"] == "incorrect"
    assert rows[0]["wrong_move_san"]


def test_sample_proposed_move_pair_items_has_correct_and_wrong_rows():
    puzzles = [
        {
            "puzzle_id": "p1",
            "fen": "k7/8/8/8/8/8/5K2/8 w - - 0 1",
            "best_move": "f2e2",
        }
    ]

    rows = list(sample_proposed_move_pair_items(puzzles, strategy="first_legal"))

    assert [row["proposed_move_condition"] for row in rows] == ["correct", "wrong"]
    assert rows[0]["proposed_move"] == "f2e2"
    assert rows[0]["expected_verdict"] == "correct"
    assert rows[1]["proposed_move"] != "f2e2"
    assert rows[1]["expected_verdict"] == "incorrect"


def test_classify_verdict_rejection_wins_over_correct_token():
    verdict = classify_verdict("Verdict: incorrect\nExplanation: this is not correct.")

    assert verdict["verdict_label"] == "rejects_proposed"
    assert verdict["flags_wrong"] is True
    assert verdict["accepts_wrong"] is False


def test_write_wrong_move_report(tmp_path):
    rows = [
        {
            "model_name": "m",
            "prompt_condition": "wrong_move_rationalize",
            "section": "pawn_endings",
            "proposed_move_condition": "wrong",
            "accepts_proposed": True,
            "rejects_proposed": False,
            "accepts_wrong": True,
            "flags_wrong": False,
            "uncertain": False,
            "verdict_label": "accepts_proposed",
            "verdict_matches_expected": False,
            "latency_ms": 100,
        },
        {
            "model_name": "m",
            "prompt_condition": "wrong_move_rationalize",
            "section": "pawn_endings",
            "proposed_move_condition": "wrong",
            "accepts_proposed": False,
            "rejects_proposed": True,
            "accepts_wrong": False,
            "flags_wrong": True,
            "uncertain": False,
            "verdict_label": "rejects_proposed",
            "verdict_matches_expected": True,
            "latency_ms": 200,
        },
    ]
    summary = tmp_path / "summary.csv"
    by_section = tmp_path / "section.csv"

    write_wrong_move_report(rows, summary, by_section)

    text = summary.read_text()
    assert "accepts_wrong_rate" in text
    assert "0.5" in text
    assert "pawn_endings" in by_section.read_text()
