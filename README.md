# Chess Reasoning Project

Research pipeline for comparing human chess reasoning and LLM chess reasoning on tactical puzzles.

## Goals
- Build a dataset of puzzles with human explanations and LLM explanations.
- Evaluate move quality, explanation structure, and "move recoverability".
- Produce paper-ready figures and tables.

## Repo Layout
```
chess_reasoning_project/
  data/
    raw/
    interim/
    processed/
  configs/
    models/
    prompts/
    experiments/
  src/
    chess_reasoning/
      ingestion/
      parsing/
      alignment/
      generation/
      evaluation/
      analysis/
      utils/
  notebooks/
  outputs/
    figures/
    tables/
    logs/
  tests/
```

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start
1) Ingest Lichess puzzles (CSV or .zst):
```
chess-reasoning ingest-lichess \
  --input data/raw/lichess_db_puzzle.csv \
  --output data/processed/puzzles.jsonl \
  --split-strategy random \
  --split-ratios 0.8,0.1,0.1 \
  --seed 42
```

1b) Sample puzzles by rating (e.g., 500-2000, 1000 puzzles):
```
chess-reasoning sample-puzzles \
  --input data/processed/puzzles.jsonl \
  --output data/processed/puzzles_500_2000_1000.jsonl \
  --min-rating 500 \
  --max-rating 2000 \
  --max-samples 1000 \
  --seed 42
```

2) Ingest PGN comments (human explanations):
```
chess-reasoning ingest-pgn-comments \
  --input data/raw/annotated_games.pgn \
  --output data/interim/human_explanations.jsonl \
  --source-url "local" \
  --license "unknown"
```

3) Match explanations to puzzles (by exact FEN):
```
chess-reasoning match-explanations \
  --puzzles data/processed/puzzles.jsonl \
  --explanations data/interim/human_explanations.jsonl \
  --output data/processed/human_explanations.jsonl
```

4) Ingest annotated moves with chess annotation symbols (NAGs) from PGN:
```
chess-reasoning ingest-pgn-annotations \
  --input data/raw/annotated_games.pgn \
  --output data/interim/move_annotations.jsonl \
  --source-url "local" \
  --license "unknown"
```

5) Score predicted annotation symbols (from your model or LLM):
```
chess-reasoning score-annotations \
  --input data/processed/annotation_predictions.jsonl \
  --output outputs/tables/annotation_scores.json
```

6) Build blind human-rating study items (explanations must include `puzzle_id`, `source_group`, and text):
```
chess-reasoning build-rating-items \
  --puzzles data/processed/puzzles.jsonl \
  --explanations data/processed/explanations_for_study.jsonl \
  --output data/processed/rating_items.jsonl \
  --sheet-output data/processed/rating_sheet.csv \
  --design single \
  --per-puzzle 2
```

7) Ingest completed ratings and analyze:
```
chess-reasoning ingest-ratings \
  --input data/processed/rating_sheet_filled.csv \
  --output data/processed/ratings.jsonl

chess-reasoning analyze-ratings \
  --items data/processed/rating_items.jsonl \
  --ratings data/processed/ratings.jsonl \
  --output outputs/tables/rating_summary.json
```

8) Generate LLM explanations with OpenAI API:
```
export OPENAI_API_KEY=your_key_here
chess-reasoning generate-openai \
  --puzzles data/processed/puzzles_500_2000_1000.jsonl \
  --prompt configs/prompts/condition_a.txt \
  --output data/processed/llm_generations.jsonl \
  --model gpt-4.1-mini \
  --temperature 0.0 \
  --max-output-tokens 256 \
  --sleep-s 0.2
```

9) Ingest book positions (manual CSV) and analyze by section:
```
chess-reasoning ingest-book \
  --input data/raw/book_positions.csv \
  --output data/processed/book_positions.jsonl \
  --source "test-your-endgame-ability"
```

10) Stockfish evaluation and section analysis:
```
chess-reasoning stockfish-eval \
  --puzzles data/processed/book_positions.jsonl \
  --generations data/processed/llm_generations.jsonl \
  --output data/processed/llm_generations_stockfish.jsonl \
  --engine-path /usr/local/bin/stockfish \
  --depth 12

chess-reasoning analyze-sections \
  --puzzles data/processed/book_positions.jsonl \
  --generations data/processed/llm_generations_stockfish.jsonl \
  --output outputs/tables/section_metrics.json
```

11) Ingest human think-aloud transcripts:
```
chess-reasoning ingest-human-transcripts \
  --input data/raw/human_transcripts.csv \
  --puzzles data/processed/book_positions.jsonl \
  --output data/processed/human_reasoning.jsonl \
  --strip-fillers
```

12) Build a unified reasoning table (LLM + human + book):
```
chess-reasoning build-reasoning-table \
  --puzzles data/processed/book_positions.jsonl \
  --llm data/processed/llm_generations_stockfish.jsonl \
  --human data/processed/human_reasoning.jsonl \
  --book data/processed/book_generations_stockfish.jsonl \
  --output data/processed/reasoning_comparison.jsonl
```

13) Compute explanation specificity features:
```
chess-reasoning explanation-specificity \
  --input data/processed/reasoning_comparison.jsonl \
  --output data/processed/reasoning_with_specificity.jsonl \
  --config configs/analysis/explanation_specificity.yaml
```

14) Run move recoverability:
```
chess-reasoning recoverability \
  --input data/processed/reasoning_with_specificity.jsonl \
  --output data/processed/recoverability_results.jsonl \
  --mask-mode strict \
  --model gpt-4.1-mini
```

15) Generate paper-ready tables:
```
chess-reasoning reasoning-report \
  --input data/processed/recoverability_results.jsonl \
  --output-dir outputs/tables/
```

16) Label Lichess puzzles by endgame section (book-style buckets):
```
chess-reasoning label-endgames \
  --input data/processed/puzzles_500_2000_1000.jsonl \
  --output data/processed/puzzles_500_2000_1000_labeled.jsonl \
  --method hybrid
```

## Notes
- This project avoids scraping. Only ingest data you are licensed to use.
- Keep provenance metadata for all human explanations.

## Next Steps
- Add LLM generation and parsing.
- Implement move recoverability masking and evaluation.
- Build analysis and plotting scripts.
- Add annotation-symbol classification experiments.
- Add human rating study aggregation plots.
