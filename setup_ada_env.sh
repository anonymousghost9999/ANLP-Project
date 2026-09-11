#!/bin/bash
# ==============================================================================
# Ada HPC Cluster (IIIT-H) Automated Environment Setup Script
# ==============================================================================

set -e

echo "======================================================================"
echo " Starting Ada Environment Setup for Continuous Prompt Scorer"
echo "======================================================================"

# 1. Load Ada CUDA & cuDNN modules
echo "[1/4] Loading CUDA 11.6 and cuDNN 8.4 modules..."
module purge || true
module load u18/cuda/11.6
module load u18/cudnn/8.4.0-cuda-11.6

# 2. Check Conda
if command -v conda &> /dev/null; then
    echo "[2/4] Conda detected at $(which conda)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda is not found in PATH or standard home directories."
    echo "Please install Miniconda to your home directory or load an appropriate module."
    exit 1
fi

# 3. Create or update conda environment
ENV_NAME="prompt_scorer"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "[3/4] Conda environment '$ENV_NAME' already exists. Activating..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "[3/4] Creating conda environment '$ENV_NAME' with Python 3.10..."
    conda create -n "$ENV_NAME" python=3.10 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

# 4. Install PyTorch with CUDA support and dependencies
echo "[4/4] Installing PyTorch with CUDA 11.8 / 11.6 support & requirements..."
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

echo "======================================================================"
echo " Verification:"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None (Run on GPU node for GPU test)\"}')"
echo "======================================================================"
echo " Setup Completed Successfully!"
echo " Activate anytime using: conda activate prompt_scorer"
echo "======================================================================"
