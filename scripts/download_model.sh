#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/models" "$ROOT/models/qwen3-0.6b-tokenizer"
GGUF="$ROOT/models/Qwen3-0.6B-Q8_0.gguf"
TOK="$ROOT/models/qwen3-0.6b-tokenizer"

if [[ ! -f "$GGUF" ]]; then
  if command -v hf >/dev/null 2>&1; then
    hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir "$ROOT/models"
  else
    curl -L --retry 5 --fail -o "$GGUF" \
      "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
  fi
fi

if [[ ! -f "$TOK/tokenizer.json" ]]; then
  if command -v hf >/dev/null 2>&1; then
    hf download Qwen/Qwen3-0.6B tokenizer.json tokenizer_config.json --local-dir "$TOK" || true
  fi
  if [[ ! -f "$TOK/tokenizer.json" ]]; then
    curl -L --retry 5 --fail -o "$TOK/tokenizer.json" \
      "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"
    curl -L --retry 5 --fail -o "$TOK/tokenizer_config.json" \
      "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json" || true
  fi
fi
ls -lh "$ROOT/models" "$TOK"
