#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.10"
INSTALL_SUBJECTIVE=0
INSTALL_COTRACKER=1

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_env.sh [options]

Create the CoVEBench evaluation environment with uv.

Options:
  --python VERSION      Python version for uv venv. Default: 3.10
  --subjective         Also install vLLM for MLLM-checklist subjective metrics.
  --no-cotracker       Skip CoTracker installation.
  -h, --help           Show this help.

Examples:
  bash scripts/setup_env.sh
  bash scripts/setup_env.sh --subjective
  bash scripts/setup_env.sh --python 3.10 --subjective
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_VERSION="${2:?Missing value for --python}"
      shift 2
      ;;
    --subjective)
      INSTALL_SUBJECTIVE=1
      shift
      ;;
    --no-cotracker)
      INSTALL_COTRACKER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: uv is not available in PATH.

Install uv first:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then restart your shell or add uv to PATH and rerun this script.
EOF
  exit 1
fi

cd "$ROOT"

echo "[1/4] Creating uv environment: .venv (prompt=covebench, python=${PYTHON_VERSION})"
uv venv --python "$PYTHON_VERSION" --seed --prompt covebench

echo "[2/4] Installing CoVEBench objective dependencies"
uv sync --extra cuda

if [[ "$INSTALL_COTRACKER" -eq 1 ]]; then
  echo "[3/4] Installing CoTracker"
  uv pip install "git+https://github.com/facebookresearch/co-tracker.git"
else
  echo "[3/4] Skipping CoTracker"
fi

if [[ "$INSTALL_SUBJECTIVE" -eq 1 ]]; then
  echo "[4/4] Installing subjective MLLM dependency: vLLM"
  uv pip install "vllm>=0.8.0"
else
  echo "[4/4] Skipping vLLM. Re-run with --subjective if you need UAS/IFS/VRS/SEM."
fi

cat <<EOF

Environment ready.

Activate it with:
  source "$ROOT/.venv/bin/activate"

Download objective weights with:
  HF_ENDPOINT=https://hf-mirror.com uv run scripts/download_weights.py

Run objective metrics with:
  uv run scripts/run_model.py --source-dir data/source --edited-dir data/my_model --checklist data/checklist.json --output-csv outputs/my_model_scores.csv

EOF
