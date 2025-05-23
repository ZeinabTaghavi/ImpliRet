#!/usr/bin/env bash
#
# start_vllm.sh  –  Launch a local vLLM OpenAI‑compatible server
#                  and export VLLM_PID so callers can shut it down.
# --------------------------------------------------------------------
# Usage:
#   bash start_vllm.sh [PORT] [MAX_LEN] [TP] [GPU_UTIL]
#
#   PORT      – HTTP port to bind (default 8000)
#   MAX_LEN   – max context window in tokens (default 16384)
#   TP        – tensor parallel size (default 4)
#   GPU_UTIL  – GPU memory utilisation fraction (default 0.95)
#
# The script honours HF_HOME.  If HF_HOME is unset it falls back to
# ~/.cache/huggingface.  Model weights are cached (or re‑used) there.
#
# --------------------------------------------------------------------

set -euo pipefail

# Ensure the correct GPUs are visible for vLLM
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
  echo "[start_vllm] CUDA_VISIBLE_DEVICES not set; defaulting to $CUDA_VISIBLE_DEVICES"
else
  echo "[start_vllm] CUDA_VISIBLE_DEVICES pre-set to $CUDA_VISIBLE_DEVICES"
fi

PORT="${1:-8000}"
MAX_LEN="${2:-16384}"
TP="${3:-4}"
GPU_UTIL="${4:-0.95}"

MODEL="meta-llama/Llama-3.3-70B-Instruct"

echo "[start_vllm] Starting vLLM server on port ${PORT}"

python -m vllm.entrypoints.openai.api_server \
       --model "$MODEL" \
       --download-dir "$HF_HOME" \
       --tensor-parallel-size "$TP" \
       --max-model-len "$MAX_LEN" \
       --gpu-memory-utilization "$GPU_UTIL" \
       --host 0.0.0.0 --port "$PORT" \
       --uvicorn-log-level info &

export VLLM_PID=$!
echo "[start_vllm] vLLM PID: $VLLM_PID"
echo "$VLLM_PID" > /tmp/vllm_server.pid

# ------------------------------------------------------------------
# Wait for vLLM /health endpoint to return HTTP 200 (engine ready)
# ------------------------------------------------------------------
echo "[start_vllm] Waiting for vLLM /health…"
until curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" | grep -q "^200$"; do
  sleep 1
  echo "[start_vllm] Waiting for vLLM /health…"
done
echo "[start_vllm] vLLM is ready!"

# Give the engine time to load weights before clients hit it
# (sleep 60 removed; health-check loop replaces fixed sleep)
