#!/usr/bin/env bash
# macOS 与 Ubuntu 共用的环境准备。已有 data/db/njupt.sqlite 时不会覆盖。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OS="$(uname -s)"
if [[ "$OS" != "Darwin" && "$OS" != "Linux" ]]; then
  echo "只支持 macOS 与 Ubuntu。" >&2
  exit 1
fi

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    PY="$c"
    break
  fi
done
if [[ -z "$PY" ]]; then
  echo "需要 Python 3.10 及以上。Ubuntu: sudo apt install python3 python3-venv python3-pip" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

if ! command -v tesseract >/dev/null 2>&1; then
  if [[ "$OS" == "Darwin" ]]; then
    echo "未找到 tesseract。请执行: brew install tesseract tesseract-lang"
  else
    echo "未找到 tesseract。请执行: sudo apt install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng"
  fi
elif ! tesseract --list-langs 2>/dev/null | grep -q '^chi_sim$'; then
  if [[ "$OS" == "Darwin" ]]; then
    echo "tesseract 缺少 chi_sim。请执行: brew install tesseract-lang"
  else
    echo "tesseract 缺少 chi_sim。请执行: sudo apt install tesseract-ocr-chi-sim"
  fi
fi

if ! command -v llama-completion >/dev/null 2>&1; then
  if [[ "$OS" == "Darwin" ]]; then
    echo "未找到 llama-completion。请执行: brew install llama.cpp"
  else
    echo "未找到 llama-completion。请从 https://github.com/ggml-org/llama.cpp/releases 安装 Ubuntu 包，或设置 LLAMA_BIN。"
  fi
  echo "没有该程序时，计算题仍可回答；开放题需要模型。"
fi

if ! command -v zstd >/dev/null 2>&1; then
  if [[ "$OS" == "Darwin" ]]; then
    echo "未找到 zstd。请执行: brew install zstd"
  else
    echo "未找到 zstd。请执行: sudo apt install zstd"
  fi
fi

if command -v git >/dev/null 2>&1 && [[ -d .git ]] && command -v git-lfs >/dev/null 2>&1; then
  git lfs pull --include="data/db/njupt.sqlite.zst" || true
fi

if [[ ! -f data/db/njupt.sqlite ]]; then
  if [[ -f data/db/njupt.sqlite.zst ]] && command -v zstd >/dev/null 2>&1; then
    mkdir -p data/db
    zstd -d -f data/db/njupt.sqlite.zst -o data/db/njupt.sqlite
  else
    echo "还没有 data/db/njupt.sqlite。先安装 zstd 与 Git LFS，再执行 git lfs pull。" >&2
  fi
fi

if [[ ! -f models/Qwen3-0.6B-Q8_0.gguf ]]; then
  bash scripts/download_model.sh
fi

echo "环境已就绪。问答: python -m njuptqa answer \"问题\""
