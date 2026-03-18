from chess_reasoning.analysis.explanation_specificity import compute_specificity_features, load_specificity_config


def test_specificity_feature_counts():
    config = load_specificity_config(None)
    text = "If the king goes to c2 then b4 wins. After Kd7, Kb3."
    features = compute_specificity_features(text, config)
    assert features["square_mentions"] >= 2
    assert features["move_mentions"] >= 2
    assert features["conditional_markers"] >= 2
    assert features["line_depth"] >= 1
