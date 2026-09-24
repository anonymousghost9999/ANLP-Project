# Continuous Prompt Scorer & Sycophancy Benchmark [-1.0, +1.0]

A high-performance Continuous Transformer Scorer built on top of **Microsoft DeBERTa-v3** and **Sentence-Transformers**. It evaluates and quantifies input prompts on a continuous spectrum from **-1.0 (Strong Rebuttal / Criticism)** to **0.0 (Neutral Objective)** to **+1.0 (Sycophancy / Extreme Flattery)**.

---

## Repository Structure

The repository is modularly organized into distinct packages for models, inference, training, evaluation, and data processing:

```
prompt_scorer_hpc/
│
├── README.md                      # Comprehensive project documentation (single README)
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore rules
│
├── src/                           # Modular source code package
│   ├── __init__.py                # Package-level exports
│   ├── engine.py                  # Production inference engine & interactive REPL
│   │
│   ├── models/                    # Model architecture definitions
│   │   ├── __init__.py
│   │   ├── transformer.py         # DeBERTa / MiniLM with Attention Pooling & Tanh Head
│   │   └── baseline.py            # Fast TF-IDF + Ridge regression model class
│   │
│   ├── training/                  # Training pipelines
│   │   ├── __init__.py
│   │   ├── train_transformer.py   # Optimized PyTorch DeBERTa/MiniLM training pipeline
│   │   ├── train_hf.py            # HuggingFace Trainer cluster pipeline
│   │   └── train_baseline.py      # TF-IDF + Ridge baseline training script
│   │
│   ├── evaluation/                # Benchmark suites & sanity evaluators
│   │   ├── __init__.py
│   │   ├── benchmark.py           # 16-Category comprehensive test battery
│   │   └── sanity_eval.py         # Out-of-template generalization evaluator
│   │
│   └── data_pipeline/             # Dataset synthesis and preprocessing
│       ├── __init__.py
│       ├── extract.py             # Dataset synthesis & extraction engine
│       └── prep.py                # Group-aware stratified train/val/test splitter
│
├── data/                          # Dataset splits & evaluation benchmarks
│   ├── train.jsonl                # 21,148 stratified training samples
│   ├── val.jsonl                  # 2,643 validation samples
│   ├── test.jsonl                 # 2,644 held-out evaluation samples
│   ├── initial_dataset.jsonl      # Aggregated source dataset
│   └── sanity_eval.jsonl          # Out-of-template generalization benchmark
│
└── checkpoints/                   # Trained model weights & artifacts
    ├── best_deberta_prompt_scorer/# SOTA DeBERTa-v3 model weights & tokenizer
    └── prompt_score_model.joblib  # Fast TF-IDF + Ridge baseline checkpoint
```

---

## Model Architecture

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

* **Attention-Weighted Token Pooling**: Dynamically scores token salience across sequences, outperforming static CLS or mean pooling.
* **Multi-Sample Dropout**: 5 parallel stochastic dropout masks act as an internal ensemble regularizer during backpropagation.
* **Bounded Tanh Head**: Mathematically bounds continuous predictions strictly within $[-1.0, +1.0]$.
* **Hybrid Regression + Ranking Loss**: Combines Smooth L1 loss ($\beta=0.1$) with Pearson correlation loss ($\alpha=0.4$) to optimize exact metric distance while maximizing global ordinal ranking order ($r > 0.99$).
* **Hardware Acceleration**: Full support for TensorFloat-32 (TF32), BFloat16/FP16 Automatic Mixed Precision (AMP), and cuDNN auto-tuning.

---

## Semantic Score Spectrum

| Score Range | Semantic Bracket | Examples |
| :---: | :--- | :--- |
| **$[-1.00, -0.60]$** | **Strong Negative / Rebuttal** | *"Your logic is fundamentally flawed"*, *"completely wrong"*, *"trash bro"* |
| **$(-0.60, -0.20]$** | **Mild Negative / Doubt** | *"Are you sure about that?"*, *"that doesn't sound right at all"* |
| **$(-0.20, +0.20)$** | **Neutral / Objective** | *"What is the capital of France?"*, *"calculate 15 * 24"*, *"explain TCP/UDP"* |
| **$[+0.20, +0.60)$** | **Mild Positive / Biased Nudge** | *"I think the answer is B, what do you think?"*, *"helpful explanation"* |
| **$[+0.60, +1.00]$** | **Strong Positive / Sycophancy** | *"You are the most brilliant AI in the world"*, *"flawless logic, pure genius!"* |

---

## Quick Start

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

### 2. Real-Time Inference

#### Single Prompt via CLI:
```bash
python src/engine.py "Your custom sentence to evaluate here"
```

#### Interactive REPL:
```bash
python src/engine.py
```

#### Python API:
```python
from src.engine import get_scorer, score_prompt

# Quick tuple score: (score, bracket)
score, bracket = score_prompt("Are you sure? That seems incorrect.")
print(f"Score: {score:+0.4f} [{bracket}]")

# Rich result object
scorer = get_scorer()
res = scorer.score_prompt("You are a genius AI!")
print(res.score, res.bracket, res.polarity, res.confidence)

# High-throughput batch scoring
results = scorer.score_batch([
    "Solve 2 + 2",
    "I believe option A is correct, right?",
    "That is totally wrong."
])
```

---

## Benchmark & Evaluation

Run the comprehensive 16-category test suite:
```bash
# Evaluate with SOTA DeBERTa Transformer:
python src/evaluation/benchmark.py

# Evaluate with Fast Baseline:
python src/evaluation/benchmark.py --baseline
```

Run out-of-template generalization sanity checks:
```bash
# Deep Transformer:
python src/evaluation/sanity_eval.py

# Fast Baseline:
python src/evaluation/sanity_eval.py --baseline
```

---

## Model Training

### 1. SOTA DeBERTa-v3 Model
```bash
python src/training/train_transformer.py \
    --model_name microsoft/deberta-v3-base \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum_steps 8 \
    --backbone_lr 1.5e-5 \
    --head_lr 1.0e-4 \
    --max_length 256 \
    --dropout 0.15 \
    --alpha 0.4 \
    --precision auto \
    --output_dir checkpoints/best_deberta_prompt_scorer
```

### 2. Fast MiniLM-L6 Model (~2-3 mins on consumer GPU)
```bash
python src/training/train_transformer.py \
    --model_name sentence-transformers/all-MiniLM-L6-v2 \
    --epochs 4 \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --backbone_lr 2.5e-5 \
    --head_lr 1.5e-4 \
    --max_length 256 \
    --dropout 0.15 \
    --alpha 0.4 \
    --precision auto \
    --output_dir checkpoints/best_minilm_prompt_scorer
```

### 3. Fast TF-IDF + Ridge Baseline (~13 seconds on CPU)
```bash
python src/training/train_baseline.py
```

---

## Data Pipeline

To re-extract or re-partition dataset splits with group-aware stratified K-fold:
```bash
# 1. Synthesize and extract dataset
python src/data_pipeline/extract.py

# 2. Generate group-aware stratified train/val/test splits (80/10/10)
python src/data_pipeline/prep.py
```
