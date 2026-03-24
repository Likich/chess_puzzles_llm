#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  echo "GROQ_API_KEY is not set. Export it first."
  exit 1
fi

PUZZLES=${PUZZLES:-"data/processed/puzzles_endgames_500_2000_200.jsonl"}
SLEEP_S=${SLEEP_S:-"0.2"}
MAX_TOKENS=${MAX_TOKENS:-"256"}

declare -a MODELS=(
  "llama-3.1-8b-instant"
  "llama-3.3-70b-versatile"
  "openai/gpt-oss-120b"
  "openai/gpt-oss-20b"
  "meta-llama/llama-4-scout-17b-16e-instruct"
  "moonshotai/kimi-k2-instruct-0905"
  "qwen/qwen3-32b"
)

declare -a PROMPTS=(
  "brief:configs/prompts/condition_a.txt"
  "calc:configs/prompts/condition_calc.txt"
  "teaching:configs/prompts/condition_teaching.txt"
)

for model in "${MODELS[@]}"; do
  model_safe="${model//\//_}"
  for entry in "${PROMPTS[@]}"; do
    IFS=":" read -r condition prompt_path <<< "${entry}"
    out="data/processed/llm_generations_groq_${model_safe}_${condition}_200.jsonl"
    echo "Running ${model} | ${condition}"
    chess-reasoning generate-openai \
      --puzzles "${PUZZLES}" \
      --prompt "${prompt_path}" \
      --output "${out}" \
      --model "${model}" \
      --temperature 0.0 \
      --max-output-tokens "${MAX_TOKENS}" \
      --prompt-condition "${condition}" \
      --api-type chat \
      --base-url "https://api.groq.com/openai/v1/chat/completions" \
      --api-key-env GROQ_API_KEY \
      --sleep-s "${SLEEP_S}"
  done
done
