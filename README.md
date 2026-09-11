# Continuous Prompt Scorer & Sycophancy Benchmark [-1.0, +1.0]

A state-of-the-art Continuous Transformer Scorer built from scratch on top of **Microsoft DeBERTa-v3** and **Sentence-Transformers**. It quantifies input prompts on a continuous scale from **-1.0 (Strong Rebuttal / Criticism)** to **0.0 (Neutral Objective)** to **+1.0 (Sycophancy / Extreme Flattery)**.

---

## Architecture Highlights

```
                       Input Prompt (Tokens)
                                │
                                ▼
               Pretrained Transformer Backbone (DeBERTa-v3)
                                │
                                ▼
               Attention-Weighted Token Pooling
                                │
                                ▼
               Multi-Sample Dropout (5 Parallel Passes, p=0.15)
                                │
                                ▼
               2-Stage MLP with GELU + Bounded Tanh Head
                                │
                                ▼
               Continuous Score: [-1.0, +1.0]
```

* **Attention-Weighted Token Pooling**: Dynamically scores token salience over standard static CLS/mean pooling.
* **Multi-Sample Dropout**: 5 parallel stochastic paths act as an ensemble regularizer during training.
* **Bounded Tanh Head**: Enforces strict $[-1.0, +1.0]$ bounds mathematically.
* **Hybrid Smooth L1 + Pearson Correlation Loss**: Minimizes regression distance while directly maximizing global ranking order ($r > 0.99$).
* **Hardware Acceleration**: Automatic Mixed Precision (BFloat16 / FP16) + TensorFloat-32 for 5x–10x speedup on NVIDIA Ampere/Ada GPUs.

---

## Semantic Score Spectrum

| Score Range | Semantic Bracket | Examples |
| :---: | :--- | :--- |
| **$[-1.00, -0.60]$** | **Strong Negative / Rebuttal** | *"Your logic is fundamentally flawed"*, *"trash bro"*, *"completely wrong"* |
| **$(-0.60, -0.20]$** | **Mild Negative / Doubt** | *"Are you sure about that?"*, *"that doesn't sound right at all"* |
| **$(-0.20, +0.20)$** | **Neutral / Objective** | *"What is the capital of France?"*, *"calculate 15 * 24"*, *"explain TCP/UDP"* |
| **$[+0.20, +0.60)$** | **Mild Positive / Biased Nudge** | *"I think the answer is B, what do you think?"*, *"helpful explanation"* |
| **$[+0.60, +1.00]$** | **Strong Positive / Sycophancy** | *"You are god"*, *"absolutely flawless logic, pure genius!"*, *"truly outstanding"* |

---

## Quick Start (Local)

### 1. Installation
```bash
# Clone the repository
git clone git@github.com:anonymousghost9999/ANLP-Project-.git
cd ANLP-Project-

# Create environment and install dependencies
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Interactive Real-Time Scorer (REPL)
```powershell
python scorer_engine.py
```

### 3. Single Prompt CLI Scoring
```powershell
python scorer_engine.py "Your custom sentence to evaluate here"
```

### 4. Training Pipeline
```powershell
# Windows one-click batch
.\run_local_deberta.bat

# Or direct command:
python train_scorer.py \
    --model_name microsoft/deberta-v3-base \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum_steps 8 \
    --backbone_lr 1.5e-5 \
    --head_lr 1.0e-4 \
    --max_length 256 \
    --precision auto \
    --output_dir best_deberta_prompt_scorer
```

---

## HPC Cluster Deployment (SLURM / Ada)

To train on SLURM clusters (e.g., IIIT-H Ada):

```bash
# 1. Setup scratch environment and cache
bash setup_ada_env.sh

# 2. Submit SLURM GPU training job
sbatch job.slurm
```

---

## Evaluation Benchmark

Run the 16-category comprehensive test battery:
```powershell
python test_suite.py --transformer
```
