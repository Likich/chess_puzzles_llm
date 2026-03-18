from chess_reasoning.evaluation.stockfish_eval import eval_generations_with_stockfish


def test_stockfish_eval_handles_illegal_move(monkeypatch):
    # Fake engine to avoid requiring Stockfish in tests
    class FakeEngine:
        def analyse(self, board, limit):
            return {"pv": [board.legal_moves.__iter__().__next__()], "score": DummyScore()}
        def quit(self):
            return None

    class DummyScore:
        def pov(self, color):
            return self
        def score(self, mate_score=100000):
            return 0

    def fake_popen_uci(path):
        return FakeEngine()

    import chess.engine
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", staticmethod(fake_popen_uci))

    puzzles = {"p1": {"puzzle_id": "p1", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1"}}
    gens = iter([{"puzzle_id": "p1", "parsed_move": "a1a8"}])
    rows = list(eval_generations_with_stockfish(puzzles, gens, engine_path="/bin/false", depth=1))
    assert rows[0]["engine_best_move"] is not None
