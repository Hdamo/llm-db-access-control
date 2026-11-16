#!/usr/bin/env bash
#SBATCH --job-name=llmdb-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Uncomment and set these to your cluster's GPU partition name if one exists
# (check with: sinfo -o "%P %G"). A 7B model is much faster with a GPU.
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
#
# Usage (run from the repo root, logs/ must exist -- setup_env.sh creates it):
#   sbatch --job-name=baseline-7b cluster/run_eval.slurm baseline
#   sbatch --job-name=phase1-7b   cluster/run_eval.slurm phase1
#   sbatch --job-name=phase2-7b   cluster/run_eval.slurm phase2
#
# Each phase writes to its own default checkpoint/report file, so all three
# can be submitted and run concurrently with no collision.
#
# Crash/timeout-resumable: if this job hits the time limit before finishing,
# just resubmit the exact same sbatch command -- the JSONL checkpoint picks
# up where it left off.

set -euo pipefail

PHASE="${1:?Usage: sbatch run_eval.slurm <baseline|phase1|phase2>}"

module load python/3.13.5 2>/dev/null || module load miniconda/default
source .venv/bin/activate
export PATH="$HOME/bin:$PATH"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama_models}"

# Match this to --workers below. Higher = more concurrent Ollama requests
# actually served in parallel (not just queued) -- see
# docs/phase2/parallel-execution.md.
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1

mkdir -p logs reports

echo "Starting ollama serve..."
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
COMMON_ARGS=(--splits easy medium hard --limit 10000 --seed 42 --model "$MODEL" --workers 4)

case "$PHASE" in
  baseline) python -m src.run_full_eval_baseline "${COMMON_ARGS[@]}" ;;
  phase1)   python -m src.run_full_eval "${COMMON_ARGS[@]}" ;;
  phase2)   python -m src.run_full_eval_v2 "${COMMON_ARGS[@]}" ;;
  *) echo "Unknown phase: $PHASE (expected baseline|phase1|phase2)" >&2; exit 1 ;;
esac

kill "$OLLAMA_PID" 2>/dev/null || true
echo "Done: $PHASE"
