#!/bin/bash

echo "========================================"
echo " Starting AnimeColor RunPod Worker"
echo "========================================"

# Fail on error
set -e

# Activate environment (if needed)
# source /opt/venv/bin/activate

# Debug info
echo "Python version:"
python --version

echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"

echo "GPU Info:"
nvidia-smi || true

# Create required directories (important for container runs)
mkdir -p /workspace/inputs/lineart
mkdir -p /workspace/outputs

echo "Environment ready."

# Start worker
echo "Starting RunPod Worker..."
python handler.py