#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Setup Script for Gating Steering Project
# ============================================================================
# This script sets up the Conda environment and installs necessary dependencies.
#
# Usage:
#   bash setup.sh
# ============================================================================

# ---- Initialize conda ----
# Source conda.sh to enable conda commands in this script
CONDA_SH=""
for conda_path in "$HOME/anaconda3" "$HOME/miniconda3" "$CONDA_PREFIX/../.." "$(dirname $(dirname $(which conda 2>/dev/null) 2>/dev/null) 2>/dev/null)"; do
    if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$conda_path/etc/profile.d/conda.sh"
        break
    fi
done

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
else
    echo "Warning: conda.sh not found. Assuming 'conda' is in PATH."
fi

# ---- Create Environment ----
echo "=== Setting up environment 'steering' ==="

if ! conda info --envs | grep -q '^steering'; then
    echo "Creating Conda environment 'steering'..."
    conda create -y -n steering python=3.10
else
    echo "Environment 'steering' already exists."
fi

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate steering

# ---- Install Dependencies ----
echo "=== Installing dependencies ==="

# Upgrade pip
pip install -q --upgrade pip

# Install PyTorch (adjust version if needed)
if ! python -c "import torch; print(torch.__version__)" &>/dev/null; then
  echo "Installing PyTorch..."
  pip3 install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -q -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

# ---- Hugging Face Setup ----
echo "=== Setting up Hugging Face Cache ==="

# Set default cache location if not already set
# Users can override this by exporting HF_HOME before running the script
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

echo "HF_HOME set to: ${HF_HOME}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# Hugging Face Login
if ! python -c "import huggingface_hub; print(huggingface_hub.get_token())" &>/dev/null; then
    echo ""
    echo "Note: You may need to log in to Hugging Face to access protected models (like Llama-3)."
    echo "Run: 'huggingface-cli login'"
fi

echo ""
echo "=== Setup Complete! ==="
echo "To activate the environment, run:"
echo "  conda activate steering"
