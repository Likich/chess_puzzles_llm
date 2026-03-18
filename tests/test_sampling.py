from chess_reasoning.ingestion.sample import reservoir_sample


def test_reservoir_sample_size():
    rows = [{"i": i} for i in range(100)]
    sample = reservoir_sample(rows, k=10, seed=1)
    assert len(sample) == 10
