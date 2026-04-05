#!/bin/bash

echo "========================================"
echo " Starting AnimeColor Worker"
echo "========================================"

set -e

echo "Python version:"
python --version

echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "GPU Info:"
nvidia-smi || true

mkdir -p /workspace/inputs/lineart
mkdir -p /workspace/outputs

echo "Environment ready."

echo "Starting Worker..."
python worker.py