from chess_reasoning.evaluation.masking import mask_explanation


def test_masking_uci_and_san():
    text = "Best move is Qh7+ or e2e4"
    masked = mask_explanation(text, level="light")
    assert "Qh7" not in masked
    assert "e2e4" not in masked


def test_masking_strict_phrases():
    text = "A queen sacrifice on h7 wins immediately."
    masked = mask_explanation(text, level="strict")
    assert "sacrifice on h7" not in masked.lower()


def test_masking_piece_to_square():
    text = "Moving the king to c2 is the idea."
    masked = mask_explanation(text, level="strict")
    assert "king to c2" not in masked.lower()
