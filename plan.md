# Execution Plan: Multi-Turn Sycophancy & Confidence Drift Evaluation on JarvisLabs

This plan outlines the end-to-end execution workflow for running the multi-turn confidence drift experiments on **JarvisLabs.ai** using **open-source LLMs** (e.g., Llama-3.1-8B-Instruct, Mistral-7B-Instruct, or Qwen-2.5-7B/14B) with **target-token log-probability confidence extraction**.

---

## 1. Experimental Methodology Summary

### 1.1 Objective & Novelty Framing
* **Prior Literature**: Perez et al. (2022) and Sharma et al. (2023) established single-turn sycophancy and single-rebuttal archetypes.
* **Our Gap/Contribution**: Extending those pressure archetypes into a **sustained, multi-turn dynamic conversation** on 75 ground-truth real-analysis claims to measure dynamic belief drift, hysteresis, and recanting thresholds.

### 1.2 Target-Token Log-Probability Confidence (Pedapati et al. 2024)
Rather than using sequence-level perplexity (which reflects linguistic fluency rather than epistemic certainty), confidence is computed via **target-token log probabilities** on the model's verdict logits:
1. At the conclusion of turn $t$, prompt the model to generate its verdict prefix:
   $$\text{Prompt Suffix: } \text{"\nConclusion: The statement is strictly [Verdict: "}$$
2. Extract the unnormalized logits for candidate tokens:
   $$z_{\text{valid}} = \text{Logit}(\text{"Valid"}), \quad z_{\text{invalid}} = \text{Logit}(\text{"Invalid"})$$
3. Compute the normalized binary softmax confidence:
   $$P_t(\text{Valid}) = \frac{\exp(z_{\text{valid}})}{\exp(z_{\text{valid}}) + \exp(z_{\text{invalid}})}$$
4. (Optional metric) Log-odds difference: $\Delta z_t = z_{\text{valid}} - z_{\text{invalid}}$.

### 1.3 Breaking the Turn-Pressure Collinearity
To prevent turn position $t$ (conversational fatigue, context growth) from confounding cumulative pressure $S_t$:
* **Per-Claim Randomization**: Within each track (positive or negative), the 6 pressure prompts are randomly permuted for each claim using a deterministic claim-specific seed ($\text{seed} = 42 + \text{claim\_id}$).
* **Tracking Variables**:
  * Instantaneous prompt score $s_{i,t} \in [-1.0, +1.0]$
  * Cumulative pressure score $S_{i,t} = \sum_{k=1}^t s_{i,k}$
  * Conversational turn $t \in [0, 6]$

---

## 2. Infrastructure & Model Selection on JarvisLabs

### 2.1 Recommended Hardware Configuration
| Model | Recommended GPU | VRAM | Runtime (75 claims, ~1,050 turns) |
| :--- | :--- | :--- | :--- |
| **Llama-3.1-8B-Instruct** | 1x RTX 4090 / A5000 | 24 GB | ~15–25 mins (HF) / ~5 mins (vLLM) |
| **Qwen-2.5-7B-Instruct** | 1x RTX 4090 | 24 GB | ~15–20 mins (HF) / ~4 mins (vLLM) |
| **Qwen-2.5-14B-Instruct** | 1x A6000 / A100 (40GB) | 48 GB | ~30–40 mins |
| **Llama-3.1-70B-Instruct (4-bit)** | 1x A100 (80GB) | 80 GB | ~1 hour |

*Recommendation*: Start with **1x RTX 4090 (24GB)** running **`meta-llama/Meta-Llama-3.1-8B-Instruct`** or **`Qwen/Qwen2.5-7B-Instruct`** using PyTorch bfloat16.

---

## 3. Step-by-Step Execution Plan

```
[Phase 1: Setup] ──> [Phase 2: Validation] ──> [Phase 3: Production Run] ──> [Phase 4: Analysis] ──> [Phase 5: Sync]
 JarvisLabs Instance    Smoke test 2 claims       All 75 claims (both tracks)     Plots & Mixed Models    Download data
```

### Phase 1: Environment Provisioning on JarvisLabs
1. **Launch Instance**:
   * Template: **PyTorch 2.x** with CUDA 12.x.
   * GPU: RTX 4090 (24GB VRAM) or A6000 (48GB).
   * Storage: 50GB SSD.
2. **Access Terminal**:
   * Connect via SSH or open the JupyterLab Terminal.
3. **Clone Repository & Sync Files**:
   ```bash
   git clone <YOUR_REPO_URL> ANLP-Project
   cd ANLP-Project
   ```
4. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install torch transformers accelerate datasets pandas matplotlib seaborn tqdm scipy statsmodels
   ```
5. **Hugging Face Authentication** (if using gated models like Llama 3):
   ```bash
   huggingface-cli login
   # Paste your HF read token with access to Meta-Llama-3
   ```

---

### Phase 2: Smoke Test & Sanity Calibration (2 Claims)
Before launching all 75 claims, run a fast verification pass on claims 1 & 2:
1. Verify `data/claims/claims_prompts.jsonl` exists and parses properly.
2. Run `run_claims_experiment.py` with `--limit_claims 2`.
3. Check generated logs:
   * Verify that logits for token `"Valid"` and `"Invalid"` are non-zero and differentiating.
   * Confirm turn indices $t=0 \dots 6$ record properly.
   * Confirm prompt orders are permuted between Claim 1 and Claim 2.

```bash
python src/evaluation/run_claims_experiment.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --limit_claims 2 \
  --output_dir results/smoke_test
```

---

### Phase 3: Full Production Run (75 Claims)
Run the full experiment across all 75 claims and both directional pressure tracks:
* Total claims: 75
* Tracks per claim: 2 (Positive track & Negative track)
* Turns per track: 7 (Turn 0 neutral baseline + 6 follow-up turns)
* Total evaluated generation steps: $75 \times 2 \times 7 = 1,050$ turns.

```bash
# Run in background with nohup to protect against disconnects
nohup python src/evaluation/run_claims_experiment.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --prompts_path data/claims/claims_prompts.jsonl \
  --output_dir results/llama3_production \
  --max_new_tokens 512 \
  > experiment.log 2>&1 &

# Monitor progress:
tail -f experiment.log
```

---

### Phase 4: Statistical Analysis & Visualization
Run the analysis pipeline to generate the core publication figures and statistical tests:

1. **Figure 1: Mean Confidence Trajectory vs. Turn $t$**:
   * Compares positive vs. negative pressure trajectories with shaded standard error ($\pm 1 \text{SE}$).
2. **Figure 2: Confidence vs. Cumulative Pressure $S_t$ (Phase Plot)**:
   * Displays the continuous non-parametric LOWESS curve across the spectrum $[-5.0, +5.0]$.
3. **Statistical Modeling (Mixed-Effects Linear Model)**:
   * Fit formula:
     $$\text{TargetConfidence}_{i,t} = \beta_0 + \beta_{\text{turn}} \cdot t + \beta_{\text{step}} \cdot s_{i,t} + \beta_{\text{cum}} \cdot S_{i,t} + (1 | \text{claim\_id})$$
   * Verifies that $\beta_{\text{cum}}$ is statistically significant ($p < 0.001$) independently of $\beta_{\text{turn}}$.

```bash
python src/evaluation/analyze_results.py \
  --results_csv results/llama3_production/claims_drift_summary.csv \
  --output_dir results/llama3_production/figures
```

---

### Phase 5: Artifact Download & Local Archival
1. Compress the results directory on JarvisLabs:
   ```bash
   tar -czvf llama3_results.tar.gz results/llama3_production/
   ```
2. Download to your local machine via SCP or JupyterLab file browser:
   ```powershell
   scp -P <PORT> root@<JARVIS_IP>:/workspace/ANLP-Project/llama3_results.tar.gz .
   ```
3. Extract locally into `ANLP-Project/results/`.
4. Shut down / destroy the JarvisLabs instance to avoid unnecessary billing.

---

## 4. Deliverables Checklist

- [ ] `src/evaluation/run_claims_experiment.py`: Production evaluation script with target-token logprob extraction and per-claim shuffle.
- [ ] `src/evaluation/analyze_results.py`: Statistical plotting and mixed-effects regression script.
- [ ] `results/llama3_production/claims_drift_results.jsonl`: Complete turn-level conversational logs.
- [ ] `results/llama3_production/claims_drift_summary.csv`: Tabular dataset ready for modeling.
- [ ] `results/llama3_production/figures/confidence_vs_turn.png`: Trajectory figure with error bands.
- [ ] `results/llama3_production/figures/confidence_vs_cum_pressure.png`: Phase plot across cumulative score $S_t$.
- [ ] `results/llama3_production/figures/regression_summary.txt`: LMM regression coefficients.

---

## 5. Codebase Architecture & File Roles

### 5.1 The Experiment Runner: `src/evaluation/run_claims_experiment.py`
**Purpose:** Core execution script run on the JarvisLabs GPU instance. It carries out the multi-turn conversational evaluation.

**Step-by-step functionality:**
1. **Loads the Open-Source Model**: Loads the specified LLM (e.g., Llama-3.1-8B, Qwen-2.5-7B) onto the GPU in `bfloat16` (or 4-bit with BitsAndBytes).
2. **Loads the Prompts Dataset**: Ingests `data/claims/claims_prompts.jsonl` and groups records by `claim_id`.
3. **Conducts Multi-Turn Conversations**:
   * For each claim, executes two independent conversational sessions: a **Positive Pressure Track** and a **Negative Pressure Track**.
   * Turn $t=0$: Feeds the neutral baseline mathematical claim/question.
   * Turns $t=1 \dots 6$: Sequentially feeds follow-up pressure prompts (flattery, doubt, rebuttal) while preserving conversational context.
4. **Shuffles Prompts (Collinearity Fix)**: Deterministically permutes the 6 prompts per track for each claim (`seed = 42 + claim_id`) so that turn position ($t$) and cumulative pressure ($S_t$) are statistically decorrelated across the dataset.
5. **Extracts Confidence via Target-Token Log Probabilities**:
   * Evaluates the next-token logits for `"Valid"` vs. `"Invalid"`:
     $$P(\text{Valid}) = \frac{\exp(z_{\text{Valid}})}{\exp(z_{\text{Valid}}) + \exp(z_{\text{Invalid}})}$$
   * Simultaneously extracts elicited verbalized confidence `[Confidence: X%]` as a secondary comparison.
6. **Saves Structured Outputs**: Streams every conversational turn to `results/.../claims_drift_results.jsonl` and saves a flattened table to `claims_drift_summary.csv`.

### 5.2 The Analysis & Plotting Engine: `src/evaluation/analyze_results.py`
**Purpose:** Post-processing script executed after the experiment finishes (either on JarvisLabs or locally).

**Step-by-step functionality:**
1. **Parses the Output Table**: Reads `claims_drift_summary.csv`.
2. **Generates Trajectory Plots**:
   * **`confidence_vs_turn.png`**: Plots the model's confidence trajectory over conversational turns $t=0 \dots 6$, comparing positive vs. negative tracks with standard error bands ($\pm 1\text{SE}$).
   * **`confidence_vs_cum_pressure.png`**: Plots model confidence against cumulative pressure $S_t$ with a non-parametric LOWESS curve.
3. **Fits Mixed-Effects Linear Models (LMM)**: Models $\text{Confidence} \sim \text{turn} + S_t$ with random intercepts grouped by `claim_id` to statistically isolate the pressure effect from turn-position decay.

### 5.3 Relationship to Existing Project Files

| File | Role in the Workflow |
| :--- | :--- |
| `data/claims/claims_prompts.jsonl` | **The Input Dataset**: 975 literature-grounded prompts (13 per claim across 75 claims) read and evaluated by `run_claims_experiment.py`. |
| `data/claims/claims.json` | **Ground Truth Source**: Contains the 75 formal real-analysis claims, proof steps, and solutions from S.K. Mapa. |
| `src/data_pipeline/build_claims_prompts.py` | **Prompt Generator**: The script that synthesized `claims_prompts.jsonl` from `claims.json` using literature-backed pressure framings. |
| `src/engine.py` | **Tonality Scorer**: The project's trained DeBERTa model that scores input prompts on $[-1.0, +1.0]$. |
| `plan.md` | **The Master Guide**: Complete step-by-step roadmap with terminal commands, hardware guidance, and methodology. |

