from chess_reasoning.analysis.endgame_labeling import label_endgame_section


def test_label_pawn_endgame():
    fen = "8/8/8/8/8/8/4k3/4K3 w - - 0 1"
    label = label_endgame_section(fen, themes=["endgame"], method="material", require_endgame_tag=True)
    assert label.section == "pawn_endings"


def test_label_rook_endgame():
    fen = "8/8/8/8/8/8/4k3/4K2R w - - 0 1"
    label = label_endgame_section(fen, themes=["endgame"], method="material", require_endgame_tag=True)
    assert label.section == "rook_endings"
