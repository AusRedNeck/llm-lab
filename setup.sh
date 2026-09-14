#!/bin/bash
# setup.sh — cross-platform setup for llm-lab
# Run after: git pull
#
# Mac: uv sync is enough (resolves correct torch wheel)
# Windows: uv sync installs CPU torch, then this fixes it to CUDA

set -e
cd "$(dirname "$0")"

echo "=== llm-lab setup ==="

# Create venv if missing
if [ ! -d .venv ]; then
    echo "Creating venv..."
    uv venv .venv --python 3.12
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Fix torch to CUDA on Windows
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Windows detected — installing CUDA torch..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
fi

# Verify
echo ""
echo "=== Verify ==="
.venv/Scripts/python.exe -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || \
.venv/bin/python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null

echo ""
echo "Done. Run with: uv run python -m train.train ..."
