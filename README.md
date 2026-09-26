# Continuous Prompt Scorer & Sycophancy Benchmark [-1.0, +1.0]

A state-of-the-art Continuous Transformer Scorer built on top of **Microsoft DeBERTa-v3-large** and **Sentence-Transformers**. It quantifies input prompts on a continuous scale from **-1.0 (Strong Rebuttal / Criticism)** to **0.0 (Neutral Objective)** to **+1.0 (Sycophancy / Extreme Flattery)**.

---

## Repository Structure

The repository is organized into dedicated modules for model architectures, inference, training pipelines, evaluation suites, and data preprocessing:

```
prompt_scorer_hpc/
│
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Project dependencies
├── .gitignore                             # Git ignore rules
│
├── src/                                   # Modular source code package
│   ├── __init__.py                        # Package-level exports
│   ├── engine.py                          # Production inference engine & interactive REPL
│   │
│   ├── models/                            # Model architecture definitions
│   │   ├── __init__.py
│   │   ├── transformer.py                 # DeBERTa-v3-large / MiniLM with Attention Pooling & Tanh Head
│   │   └── baseline.py                    # Fast TF-IDF + Ridge regression model class
│   │
│   ├── training/                          # Training pipelines
│   │   ├── __init__.py
│   │   ├── train_transformer.py           # PyTorch DeBERTa-v3-large/base training with Adafactor & TF32
│   │   ├── train_hf.py                    # HuggingFace Trainer cluster pipeline
│   │   └── train_baseline.py              # TF-IDF + Ridge baseline training script
│   │
│   ├── evaluation/                        # Benchmark suites & sanity evaluators
│   │   ├── __init__.py
│   │   ├── benchmark.py                   # 16-Category comprehensive test battery
│   │   └── sanity_eval.py                 # Out-of-template generalization evaluator
│   │
│   └── data_pipeline/                     # Dataset synthesis, curation & preprocessing
│       ├── __init__.py
│       ├── preprocess_tonality.py         # Curated 20k tonality dataset preprocessor & splitter
│       ├── extract.py                     # Dataset synthesis & template extraction engine
│       └── prep.py                        # Group-aware stratified train/val/test splitter
│
├── data/                                  # Dataset splits & evaluation benchmarks
│   ├── curated/                           # 20k Curated tonality splits (train.jsonl, val.jsonl, test.jsonl)
│   ├── final_tonality_dataset.jsonl       # Curated tonality source dataset
│   ├── train.jsonl                        # Baseline stratified training samples
│   ├── val.jsonl                          # Baseline validation samples
│   ├── test.jsonl                         # Baseline held-out evaluation samples
│   ├── initial_dataset.jsonl              # Source dataset
│   └── sanity_eval.jsonl                  # Out-of-template generalization benchmark
│
└── checkpoints/                           # Trained model weights & artifacts
    ├── best_deberta_large_curated_scorer/ # SOTA DeBERTa-v3-large model weights & tokenizer (r = 0.9710)
    ├── best_deberta_prompt_scorer/        # DeBERTa-v3-base model weights & tokenizer
    └── prompt_score_model.joblib          # Fast TF-IDF + Ridge baseline checkpoint
```

---

## Model Architecture

```
                       Input Prompt (Tokens)
                                │
                                ▼
            Pretrained Backbone (DeBERTa-v3-large / base)
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

* **Attention-Weighted Token Pooling**: Dynamically weights token salience across sequences rather than static CLS or mean pooling.
* **Multi-Sample Dropout**: 5 parallel stochastic dropout paths act as an internal ensemble regularizer during training.
* **Bounded Tanh Head**: Mathematically bounds continuous outputs strictly within `[-1.0, +1.0]`.
* **Hybrid Regression + Ranking Loss**: Combines Smooth L1 loss ($eta=0.1$) with Pearson correlation loss ($lpha=0.5$) to optimize regression distance while maximizing global ordinal ranking order ($r > 0.97$).
* **Adafactor Optimizer & Gradient Checkpointing**: Enables training large 435M-parameter DeBERTa-v3-large models on consumer/cluster GPUs (11GB VRAM).
* **Hardware Acceleration**: Automatic Mixed Precision (BFloat16 / FP16), TensorFloat-32 (TF32), and cuDNN auto-tuning.

---

## Model Performance Benchmarks

### Held-Out Unseen Test Set (2,000 Prompts)
| Metric | DeBERTa-v3-large (Curated) | DeBERTa-v3-base | TF-IDF + Ridge Baseline |
| :--- | :---: | :---: | :---: |
| **Pearson Correlation ($r$)** | **$0.9710$ ($97.10\%$)** | $0.9412$ ($94.12\%$) | $0.8953$ ($89.53\%$) |
| **Spearman Correlation ($ho$)** | **$0.9274$ ($92.74\%$)** | $0.9021$ ($90.21\%$) | $0.8766$ ($87.66\%$) |
| **$R^2$ Goodness-of-Fit** | **$0.9423$** | $0.8841$ | $0.8003$ |
| **Root Mean Squared Error (RMSE)** | **$0.1354$** | $0.1840$ | $0.2795$ |
| **Mean Absolute Error (MAE)** | **$0.0672$** | $0.1120$ | $0.2051$ |
| **3-Way Directional Accuracy** | **$95.02\%$** | $92.15\%$ | $84.13\%$ |
| **5-Way Semantic Bracket Accuracy** | **$89.69\%$** | $84.50\%$ | $74.20\%$ |
| **Out-of-Template Sanity Pearson ($r$)** | **$0.9394$** | $0.8520$ | $0.1834$ |

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
# Windows: .\.venv\Scriptsctivate
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
# Evaluate with SOTA DeBERTa-v3-large Transformer:
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

### 1. SOTA DeBERTa-v3-large Model (Curated 20k Dataset)
```bash
python src/training/train_transformer.py     --model_name microsoft/deberta-v3-large     --data_dir data/curated     --epochs 3     --batch_size 8     --grad_accum_steps 8     --backbone_lr 1.0e-5     --head_lr 8.0e-5     --max_length 256     --dropout 0.15     --alpha 0.5     --optimizer adafactor     --gradient_checkpointing     --precision auto     --output_dir checkpoints/best_deberta_large_curated_scorer
```

### 2. Fast MiniLM-L6 Model (~2-3 mins on consumer GPU)
```bash
python src/training/train_transformer.py     --model_name sentence-transformers/all-MiniLM-L6-v2     --epochs 4     --batch_size 32     --grad_accum_steps 2     --backbone_lr 2.5e-5     --head_lr 1.5e-4     --max_length 256     --dropout 0.15     --alpha 0.4     --optimizer adamw     --precision auto     --output_dir checkpoints/best_minilm_prompt_scorer
```

### 3. Fast TF-IDF + Ridge Baseline (~13 seconds on CPU)
```bash
python src/training/train_baseline.py
```

---

## Data Pipeline

To re-curate or preprocess tonality datasets with group-aware stratified K-fold:
```bash
# 1. Curate and filter 20k tonality dataset
python src/data_pipeline/preprocess_tonality.py

# 2. Extract synthetic continuous scoring records (optional)
python src/data_pipeline/extract.py
```
