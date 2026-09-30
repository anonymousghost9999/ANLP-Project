#!/usr/bin/env bash
# ==============================================================================
# One-Click Environment Setup Script for JarvisLabs.ai GPU Instances
# ==============================================================================

set -e

echo "=============================================================================="
echo " Setting up ANLP Sycophancy & Confidence Drift Environment on JarvisLabs"
echo "=============================================================================="

# 1. Upgrade pip and system packages
echo -e "\n[1/4] Upgrading pip and essential build tools..."
python3 -m pip install --upgrade pip setuptools wheel

# 2. Install PyTorch ecosystem and Hugging Face libraries
echo -e "\n[2/4] Installing PyTorch, Transformers, Accelerate, BitsAndBytes..."
python3 -m pip install \
    torch \
    torchvision \
    torchaudio \
    transformers \
    accelerate \
    bitsandbytes \
    huggingface_hub \
    datasets \
    scipy \
    statsmodels \
    pandas \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    scikit-learn

# 3. Verify CUDA GPU availability
echo -e "\n[3/4] Verifying GPU and CUDA availability..."
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:    ', torch.cuda.get_device_name(0))
    print('VRAM Available: ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
    print('bfloat16 Support:', torch.cuda.is_bf16_supported())
else:
    print('WARNING: CUDA is NOT available! GPU acceleration will not be active.')
"

# 4. Prompt for Hugging Face Login if token provided or needed
echo -e "\n[4/4] Environment setup complete!"
echo "=============================================================================="
echo " NOTE FOR GATED MODELS (e.g. Meta-Llama-3.1-8B-Instruct):"
echo " Run the following command in terminal to log in with your HF token:"
echo "   huggingface-cli login"
echo "=============================================================================="
