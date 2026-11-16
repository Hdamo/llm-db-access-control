#!/usr/bin/env bash
# One-time environment setup on the cluster LOGIN node (needs internet access
# for pip/model/dataset downloads -- compute nodes often don't have it).
# Run from the repo root: bash cluster/setup_env.sh
set -euo pipefail

echo "== Python venv =="
module load python/3.13.5 2>/dev/null || module load miniconda/default
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "== Ollama (portable binary, no sudo/root needed) =="
if ! command -v ollama >/dev/null 2>&1; then
  mkdir -p "$HOME/bin"
  curl -L https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz
  tar -xzf /tmp/ollama.tgz -C "$HOME/bin" --strip-components=1 ./bin/ollama
fi
export PATH="$HOME/bin:$PATH"

echo "== Pulling model weights (needs internet -- do this on the login node) =="
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama_models}"
mkdir -p "$OLLAMA_MODELS"
nohup ollama serve >"$HOME/ollama_serve_setup.log" 2>&1 &
OLLAMA_PID=$!
sleep 5
ollama pull qwen2.5:7b-instruct
kill "$OLLAMA_PID" 2>/dev/null || true

echo "== Caching the OrgAccess dataset (needs internet) =="
python -m src.load_orgaccess

mkdir -p logs reports

cat <<EOF

Setup done.
  OLLAMA_MODELS = $OLLAMA_MODELS   (weights live here -- compute nodes need this same path)
  HF cache      = ~/.cache/huggingface  (dataset -- compute nodes need this same \$HOME)

If \$HOME is NOT shared between login and compute nodes on this cluster,
copy both of those directories to wherever your compute nodes DO see
(scratch/shared storage), and export OLLAMA_MODELS accordingly in
cluster/run_eval.slurm before submitting.

Next: sbatch cluster/run_eval.slurm baseline
      sbatch cluster/run_eval.slurm phase1
      sbatch cluster/run_eval.slurm phase2
EOF
