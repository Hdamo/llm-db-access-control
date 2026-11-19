#!/usr/bin/env bash
#SBATCH --job-name=gepa-baseline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=cpu-long
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
# Deliberately NO --gres=gpu=... -- this account's GrpTRES caps the WHOLE
# account at 2 GPUs total (see `sacctmgr show assoc`), and both are usually
# already in use by other jobs. Requesting a GPU here, on ANY partition,
# joins that same 2-GPU queue and can delay other GPU jobs (e.g. phase2).
# Running this on CPU instead means it never competes for that cap at all --
# slower per call, but GEPA's own optimization budget is small (tens of
# examples x ~200 metric calls), unlike the 10k-row eval runs.
#
# Usage (run from the repo root, logs/ must exist -- setup_env.sh creates it):
#   sbatch cluster/run_gepa_optimize.slurm                      # defaults: 40 train / 20 val / 200 metric calls
#   sbatch cluster/run_gepa_optimize.slurm 20 10 100            # smaller/faster: 20 train / 10 val / 100 metric calls
#
# Output (written to reports/, readable over plain SSH, no Excel/GUI needed):
#   reports/gepa_best_baseline_prompt.txt    -- the winning system prompt text
#   reports/gepa_best_baseline_summary.json  -- seed vs. best val score, call counts

set -euo pipefail

TRAIN_LIMIT="${1:-40}"
VAL_LIMIT="${2:-20}"
MAX_METRIC_CALLS="${3:-200}"

module load python/3.13.5 2>/dev/null || module load miniconda/default
source .venv/bin/activate
export PATH="$HOME/ollama-dist/bin:$PATH"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama_models}"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

mkdir -p logs reports

echo "Starting ollama serve (CPU-only node -- no GPU requested)..."
nohup ollama serve >"logs/ollama_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

echo "Waiting for ollama to be ready..."
ready=0
for i in $(seq 1 30); do
  if curl -sf http://localhost:11434/ >/dev/null; then
    ready=1
    echo "ollama ready after ${i}x2s"
    break
  fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  echo "ollama server never became ready -- aborting" >&2
  kill "$OLLAMA_PID" 2>/dev/null || true
  exit 1
fi

MODEL="qwen2.5:7b-instruct"
echo "Models visible to this ollama server:"
ollama list || true
if ! ollama list | grep -q "$MODEL"; then
  echo "$MODEL not visible to this node's ollama server -- attempting to pull it here"
  ollama pull "$MODEL"
fi

python -m src.gepa_optimize_baseline \
  --train-limit "$TRAIN_LIMIT" --val-limit "$VAL_LIMIT" --max-metric-calls "$MAX_METRIC_CALLS"

kill "$OLLAMA_PID" 2>/dev/null || true
echo "Done: gepa-baseline"
