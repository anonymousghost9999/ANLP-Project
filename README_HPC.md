# Continuous Prompt Scorer on Ada HPC Cluster (IIIT Hyderabad)

This repository contains the dataset, pipelines, training scripts, and SLURM configurations configured for the **Ada HPC Cluster** at IIIT Hyderabad.

---

## 1. Quick Cluster Specifications (Ada)
- **GPU Compute Nodes**:
  - `gnodes[01-40]`: 4x NVIDIA GeForce GTX 1080 Ti GPUs (11 GB VRAM each)
  - `gnodes[43-92]`: 4x NVIDIA GeForce RTX 2080 Ti GPUs (11 GB VRAM each)
- **Partitions**: `long` (standard training), `short` (compilation & quick debug, max 6 hrs)
- **Mandatory Policy**: **1 GPU : 10 CPUs** with `2G` memory per CPU (20GB total per GPU). CVIT/HPC admins terminate jobs violating this ratio without warning.
- **Storage**:
  - `/home/$USER`: 25 GB quota (backed up daily; ideal for code and virtual environment).
  - `/scratch/$USER`: Fast local scratch on compute nodes (auto-configured by our SLURM script for HuggingFace model cache).

---

## 2. Step 1: Transfer Code to Ada

From your local machine (with IIIT VPN active if off-campus):

```bash
# Using rsync (Recommended - resumable):
rsync -avz --exclude='*.joblib' --exclude='fine_tuned_prompt_scorer' ./ username@ada.iiit.ac.in:~/prompt_scorer_hpc/

# Or using scp:
scp -r ./ username@ada.iiit.ac.in:~/prompt_scorer_hpc/
```

Then SSH into the Ada login node:
```bash
ssh -X username@ada.iiit.ac.in
cd ~/prompt_scorer_hpc
```

---

## 3. Step 2: One-Time Environment Setup on Ada

Run our automated setup script on the login node:

```bash
bash setup_ada_env.sh
```

This script automatically:
1. Loads cluster modules: `u18/cuda/11.6` and `u18/cudnn/8.4.0-cuda-11.6`.
2. Creates a dedicated Conda environment named `prompt_scorer` with Python 3.10.
3. Installs PyTorch with CUDA acceleration and all dependencies (`transformers`, `datasets`, `scikit-learn`, `scipy`, `pandas`).

---

## 4. Step 3: Train Models on Ada

### Option A: Deep Transformer Fine-Tuning on GPU (SLURM Batch Job)

Submit the GPU training job to the `long` partition:

```bash
sbatch job.slurm
```

> **Account customization**: By default, `job.slurm` uses `-A research` and `--qos=medium`. If your account is affiliated with CVIT, NLP, or another lab, edit the `#SBATCH -A` and `#SBATCH --qos=` headers in `job.slurm` (e.g. `#SBATCH -A $USER` and `#SBATCH --qos=normal`).

#### Monitor Training in Real Time:
```bash
# Check queue status:
squeue -u $USER

# Stream live training loss and evaluation metrics:
tail -f logs_*.out
```

---

### Option B: Interactive GPU Session for Debugging

To start an interactive terminal on a GPU compute node:

```bash
bash interactive_gpu.sh
# Or manually:
# srun --pty --partition=long -A research --gres=gpu:1 --mem-per-cpu=2G -c 10 bash -l
```

Once allocated:
```bash
conda activate prompt_scorer
python train_transformer_hpc.py --model_name microsoft/deberta-v3-base --epochs 3 --batch_size 32
```

---

### Option C: Instant CPU Baseline (Ridge + N-grams, ~13 seconds)

To run the fast CPU baseline:

```bash
# Directly in active shell:
conda activate prompt_scorer
python train_classifier.py

# Or via SLURM:
sbatch job_baseline.slurm
```

---

## 5. Step 4: Testing & Inference

### Score Custom Prompts with Fine-Tuned Transformer:
```bash
conda activate prompt_scorer

# Single prompt score:
python predict_transformer.py "Are you sure? I think you made a mistake in line 3."

# Interactive REPL:
python predict_transformer.py
```

### Run Comprehensive Benchmark Battery:
```bash
# Evaluate with Fast Ridge Baseline:
python test_suite.py

# Evaluate with Fine-Tuned Transformer:
python test_suite.py --transformer
```

---

## 6. Project Structure Overview

```
prompt_scorer_hpc/
│
├── setup_ada_env.sh          # One-click environment installer for Ada
├── job.slurm                 # GPU batch submission script for Ada (1:10 CPU ratio)
├── job_baseline.slurm        # CPU baseline SLURM script
├── interactive_gpu.sh        # Helper for launching interactive GPU sessions
│
├── train_transformer_hpc.py  # HuggingFace Transformer fine-tuning (DeBERTa / MiniLM)
├── train_classifier.py       # Fast multi-granularity N-gram + Ridge baseline
├── predict_transformer.py    # Transformer inference & interactive REPL
├── predict.py                # Ridge baseline inference
├── test_suite.py             # 16-category evaluation test battery
│
├── classifier_data/
│   ├── train.jsonl           # 21,148 stratified samples [-1.0, +1.0]
│   ├── val.jsonl             # 2,643 validation samples
│   └── test.jsonl            # 2,644 test evaluation samples
│
├── requirements.txt          # Python package requirements
├── README_HPC.md             # This guide
└── ada.md                    # Official Ada HPC user manual reference
```
