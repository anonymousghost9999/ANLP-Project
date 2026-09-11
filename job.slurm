#!/bin/bash
# ==============================================================================
# Ada HPC Cluster (IIIT-H) SLURM Batch Submission Script
# Task: Fine-Tune Continuous Prompt Scorer (Transformer Regressor)
# ==============================================================================

# --- 1. SLURM Directives ---
# Account: 'research' (default for all users). If affiliated with CVIT/NLP, use your account/username.
#SBATCH -A research
# QoS: 'medium' for research account; 'normal' for center accounts (cvit, nlp, etc.)
#SBATCH --qos=medium

# Partition: 'long' (gnodes 03-92 with GTX 1080 Ti & RTX 2080 Ti GPUs)
#SBATCH --partition=long

# Job Name & Real-time Logs
#SBATCH --job-name=prompt_scorer
#SBATCH --output=logs_%j.out
#SBATCH --error=logs_%j.err

# Resource Allocation: STRICT 1:10 GPU-to-CPU ratio as required by Ada policies!
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G

# Wall Time limit (HH:MM:SS)
#SBATCH --time=24:00:00

# ==============================================================================
# --- 2. Environment & Modules Setup ---
# ==============================================================================
echo "======================================================================"
echo "Job Started at: $(date)"
echo "Running on node: $(hostname)"
echo "Assigned GPU(s): $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# Load Ada CUDA & cuDNN modules
module purge
module load u18/cuda/11.6
module load u18/cudnn/8.4.0-cuda-11.6

# Set Python to unbuffered mode so SLURM logs update in real time
export PYTHONUNBUFFERED=1

# Configure Hugging Face cache in compute node local /scratch to protect 25GB /home quota
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
    export TRANSFORMERS_CACHE="/scratch/$USER/hf_cache"
    mkdir -p "$HF_HOME"
    echo "HuggingFace cache set to: $HF_HOME"
fi

# Activate Conda Environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

conda activate prompt_scorer || echo "Warning: Conda env 'prompt_scorer' activation failed. Using active Python."

# Verify environment and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# ==============================================================================
# --- 3. Run Fine-Tuning ---
# ==============================================================================
# Choose either MiniLM (Fast & lightweight) or DeBERTa-v3 (State-of-the-art):
# MODEL="sentence-transformers/all-MiniLM-L6-v2"
MODEL="microsoft/deberta-v3-base"

echo "Launching transformer fine-tuning with model: $MODEL"

python train_scorer.py \
    --model_name "$MODEL" \
    --epochs 3 \
    --batch_size 32 \
    --backbone_lr 2e-5 \
    --head_lr 1e-4 \
    --max_length 256 \
    --output_dir best_prompt_scorer

echo "======================================================================"
echo "Job Finished at: $(date)"
echo "======================================================================"
