#!/usr/bin/env bash
set -euo pipefail

# Ollama API の接続先
UBUNTU_IP="127.0.0.1"        # 同じマシンで動かすので localhost でOK
PORT="11434"
BASE_URL="http://${UBUNTU_IP}:${PORT}"

# ---- 1. サーバが生きているか簡易チェック（関連 fetch）----
# /api/tags を叩いて daemon 起動確認だけしておく
echo "[ollama-warmup] ウォームアップを開始します (BASE_URL=${BASE_URL})"
curl -sS "${BASE_URL}/api/tags" > /dev/null 2>&1 || exit 0

# ---- 2. モデルを pull（既にある場合は何もしない）----
ollama pull phi3:mini  > /dev/null 2>&1 || true
ollama pull llama3:8b   > /dev/null 2>&1 || true

# ---- 3. /api/generate で各モデルをウォームアップ ----
warm_model () {
  local model="$1"

  echo "[ollama-warmup] モデル '${model}' のウォームアップを開始します"
  curl -sS -X POST "${BASE_URL}/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "'"${model}"'",
      "prompt": "warm up",
      "stream": false
    }' \
    > /dev/null 2>&1 || true

  echo "[ollama-warmup] モデル '${model}' のウォームアップが完了しました"
}

warm_model "phi3:mini"
warm_model "llama3:8b"

echo "[ollama-warmup] すべてのウォームアップが完了しました"
