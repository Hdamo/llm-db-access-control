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

echo "== Ollama (portable, no sudo/root needed) =="
# Ollama's Linux release asset is a .tar.zst (zstd, NOT gzip) archive
# containing a bin/ + lib/ directory pair -- keep them as siblings, the
# ollama binary finds its backend libs relative to its own path.
if [ ! -x "$HOME/ollama-dist/bin/ollama" ]; then
  mkdir -p "$HOME/ollama-dist"
  curl -fL https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst \
    -o /tmp/ollama.tar.zst
  if tar --zstd -tf /tmp/ollama.tar.zst >/dev/null 2>&1; then
    tar --zstd -xf /tmp/ollama.tar.zst -C "$HOME/ollama-dist"
  else
    echo "system tar lacks built-in zstd support -- falling back to Python"
    pip install --quiet zstandard
    python3 - <<'PYEOF'
import os, tarfile, zstandard
home = os.environ["HOME"]
with open("/tmp/ollama.tar.zst", "rb") as f:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            tar.extractall(path=os.path.join(home, "ollama-dist"))
PYEOF
  fi
fi
export PATH="$HOME/ollama-dist/bin:$PATH"
ollama --version

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
