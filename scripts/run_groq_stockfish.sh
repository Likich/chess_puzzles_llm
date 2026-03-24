#!/usr/bin/env bash
set -euo pipefail

PUZZLES=${PUZZLES:-"data/processed/puzzles_endgames_500_2000_200.jsonl"}
ENGINE_PATH=${ENGINE_PATH:-"/opt/homebrew/bin/stockfish"}
DEPTH=${DEPTH:-"12"}

for gen in data/processed/llm_generations_groq_*_200.jsonl; do
  if [[ ! -f "$gen" ]]; then
    continue
  fi
  out="${gen%.jsonl}_stockfish.jsonl"
  echo "Stockfish eval: ${gen}"
  chess-reasoning stockfish-eval \
    --puzzles "${PUZZLES}" \
    --generations "${gen}" \
    --output "${out}" \
    --engine-path "${ENGINE_PATH}" \
    --depth "${DEPTH}"
done
