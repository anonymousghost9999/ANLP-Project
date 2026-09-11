#!/bin/bash
# ==============================================================================
# Ada HPC Cluster (IIIT-H) SLURM Batch Submission Script
# Task: Fast CPU Baseline Training (N-gram TF-IDF + Ridge Regressor)
# ==============================================================================

#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=short
#SBATCH --job-name=prompt_scorer_baseline
#SBATCH --output=logs_baseline_%j.out
#SBATCH --error=logs_baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00

echo "======================================================================"
echo "Job Started at: $(date)"
echo "Running on node: $(hostname)"
echo "======================================================================"

export PYTHONUNBUFFERED=1

# Activate Conda Environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate prompt_scorer || echo "Using active Python."

echo "Training fast N-gram Ridge baseline on CPU..."
python train_classifier.py

echo "======================================================================"
echo "Job Finished at: $(date)"
echo "======================================================================"
