#!/usr/bin/env bash
#SBATCH --job-name=llmdb-eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-1day
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Partition cheat sheet for this cluster (from `sinfo -o "%P %l %D"`):
#   gpu-short  8:00:00   gpu-long  12:00:00   gpu-1day  1-00:00:00   gpu-3day  3-00:00:00
# Override at submit time, e.g. for a quick smoke test:
#   sbatch --partition=gpu-short --time=1:00:00 --job-name=smoke cluster/run_eval.slurm baseline 50
#
# Usage (run from the repo root, logs/ must exist -- setup_env.sh creates it):
#   sbatch --job-name=baseline-7b cluster/run_eval.slurm baseline
#   sbatch --job-name=phase1-7b   cluster/run_eval.slurm phase1
#   sbatch --job-name=phase2-7b   cluster/run_eval.slurm phase2
# Optional 2nd arg overrides the row limit (default 10000) -- use a small
# number for a smoke test before committing a full GPU time slot:
#   sbatch --partition=gpu-short --job-name=smoke cluster/run_eval.slurm baseline 50
#
# Each phase writes to its own default checkpoint/report file, so all three
# can be submitted and run concurrently with no collision.
#
# Crash/timeout-resumable: if this job hits the time limit before finishing,
# just resubmit the exact same sbatch command -- the JSONL checkpoint picks
# up where it left off.
#
# To run a variant WITHOUT clobbering the default checkpoint/report (e.g. a
# GEPA-evolved-prompt comparison run), set CHECKPOINT/REPORT before sbatch:
#   CHECKPOINT=reports/full_eval_baseline_gepa_checkpoint.jsonl \
#   REPORT=reports/full_eval_baseline_gepa_report.xlsx \
#   LLMDB_BASELINE_SYSTEM_PROMPT_FILE=reports/gepa_best_baseline_prompt.txt \
#     sbatch --job-name=baseline-gepa cluster/run_eval.slurm baseline 2000

set -euo pipefail

PHASE="${1:?Usage: sbatch run_eval.slurm <baseline|phase1|phase2> [limit]}"
LIMIT="${2:-10000}"

module load python/3.13.5 2>/dev/null || module load miniconda/default
source .venv/bin/activate
export PATH="$HOME/ollama-dist/bin:$PATH"
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
COMMON_ARGS=(--splits easy medium hard --limit "$LIMIT" --seed 42 --model "$MODEL" --workers 4)

echo "Checking GPU visibility..."
nvidia-smi -L || echo "nvidia-smi not found / no GPU visible -- Ollama will fall back to CPU"

echo "HOME=$HOME  OLLAMA_MODELS=$OLLAMA_MODELS"
echo "Models visible to this ollama server:"
ollama list || true

if ! ollama list | grep -q "$MODEL"; then
  echo "$MODEL not visible to this node's ollama server -- attempting to pull it here"
  echo "(this needs outbound internet from the compute node; if it fails, the"
  echo " model directory likely isn't shared between login and compute nodes --"
  echo " copy \$HOME/ollama_models from the login node to shared/scratch storage instead)"
  ollama pull "$MODEL"
fi

case "$PHASE" in
  baseline) python -m src.run_full_eval_baseline "${COMMON_ARGS[@]}" ;;
  phase1)   python -m src.run_full_eval "${COMMON_ARGS[@]}" ;;
  phase2)   python -m src.run_full_eval_v2 "${COMMON_ARGS[@]}" ;;
  *) echo "Unknown phase: $PHASE (expected baseline|phase1|phase2)" >&2; exit 1 ;;
esac

kill "$OLLAMA_PID" 2>/dev/null || true
echo "Done: $PHASE"
