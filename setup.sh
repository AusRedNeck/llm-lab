#!/bin/bash
# setup.sh — cross-platform setup for llm-lab
# Run after: git pull
#
# Mac: uv sync is enough (resolves correct torch wheel)
# Windows: use system Python for training (has CUDA torch)
#          or run setup.sh to create an isolated venv for dev/test

set -e
cd "$(dirname "$0")"

echo "=== llm-lab setup ==="

# Create venv if missing (for dev/test, not training)
if [ ! -d .venv ]; then
    echo "Creating venv..."
    uv venv .venv --python 3.12
fi

# Sync dependencies
echo "Syncing dependencies..."
uv sync

# Fix torch to CUDA on Windows (overrides lockfile's CPU-only resolution)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Windows detected — installing CUDA torch..."
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
    echo ""
    echo "NOTE: 'uv run' re-syncs from lockfile and may undo this."
    echo "For training, use system Python: python -m train.train ..."
    echo "For dev/test, use: .venv/Scripts/python.exe ..."
fi

# Verify
echo ""
echo "=== Verify ==="
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    .venv/Scripts/python.exe -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null
    echo "System Python:"
    python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null
else
    .venv/bin/python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null
fi

echo ""
echo "Done."
