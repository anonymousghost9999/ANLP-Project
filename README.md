# Impact of Tonality on LLM Confidence in a Multi-Turn Setting

ANLP 2026 Monsoon project. We study what happens to a language model's verdict and confidence when a user keeps **criticising** or keeps **praising** it, on mathematical claims whose truth is known, and whether the effect can be reversed.

The proposal is in [`docs/anlp_team-Proposal.pdf`](docs/anlp_team-Proposal.pdf).

---

## Status (2 Oct 2026)

| Objective | What | Status |
| :--- | :--- | :--- |
| Data | 75 real-analysis claims and 75 false twins | **Done.** An audit found 5 originals false as written; they are corrected. Twins are LLM-drafted and **not yet human-verified**. |
| Tonality scorer | Continuous valence score for feedback turns | Trained; **labels not yet validated** (see [Scorer](#tonality-scorer)). Not used in the experiments yet. |
| O1 | Sustained-pressure trajectory | Fixed runner ready (control track, no early stopping, two probes). Pilot salvaged in `results/o1_v2_salvaged`. **Full fixed run pending.** |
| O2 | Reversibility under criticism | Runner implemented. Still pools false claims (where criticism is correct) and uses early stopping; **fix pending**. |
| O3–O6 | Recovery, confidence floor, schedule search, point of no return | Planned (see [`docs/jarvislabs_runbook.md`](docs/jarvislabs_runbook.md) for the timeline). O4's turns-below-floor is already logged by the runners. |

---

## Repository layout

```
ANLP-Project-/
├── README.md
├── requirements.txt
├── docs/
│   ├── anlp_team-Proposal.pdf
│   ├── jarvislabs_runbook.md        # How to run experiments on JarvisLabs, plus the timeline
│   ├── claims_dataset.md            # Claims, false twins and prompts: how they were built
│   └── repo_audit_2026-10.md        # Audit notes from 1 Oct 2026 (historical)
├── scripts/
│   ├── setup_jarvislabs.sh          # Install dependencies on a JarvisLabs instance
│   ├── download_model.py            # Pre-download HF model weights
│   ├── run_jarvislabs_experiment.sh # O1: praise, criticism and control tracks + analysis
│   ├── run_jarvislabs_reversibility_experiment.sh   # O2
│   ├── run_train_deberta_large.sh   # Train the tonality scorer
│   ├── verify_curated_dataset.py    # Sanity checks on data/curated
│   └── run_ada.sh                   # Notes for the IIIT Ada cluster
├── src/
│   ├── engine.py                    # Tonality scorer inference
│   ├── models/                      # Scorer architecture + TF-IDF baseline
│   ├── training/                    # Scorer training
│   ├── data_pipeline/               # Scorer data, claims, false twins, prompts
│   └── evaluation/                  # O1/O2 runners, analysis, report tables
├── data/
│   ├── claims/                      # claims.json, false twins, paired claims, prompts
│   ├── curated/                     # Scorer train/val/test splits
│   ├── raw/final_tonality_dataset.jsonl   # Scorer source dataset
│   └── sanity_eval.jsonl            # 61 hand-written out-of-template feedback turns
├── results/
│   ├── o1_full_run_v2/              # O1 pilot, raw (earlier pipeline; see caveats)
│   ├── o1_v2_salvaged/              # O1 pilot, cleaned for reporting + analysis
│   ├── o2_full_run_v2/              # O2 first run (earlier pipeline)
│   ├── scorer/                      # Scorer test-set metrics
│   └── archive/                     # Superseded or partial runs, kept for the record
├── checkpoints/                     # Scorer configs/tokenizers (weights are gitignored)
├── archive/                         # Legacy copy of the scorer preprocessing script
└── tests/
```

---

## Data

**Claims** (`data/claims/claims.json`): 75 statements from S. K. Mapa, *Introduction to Real Analysis*, across 14 chapters (68 *Hard*, 7 *Advanced*).
- Claims 16, 20, 51, 55 and 69 were false as written and have been corrected.
- Each corrected record keeps `claim_as_originally_written` and a `correction_note`.

**False twins** (`data/claims/false_twins_curated.json`): one false but plausible variant per claim, following BrokenMath (Petrov et al. 2025).
- Each twin records a `perturbation_type` and a `falsity_reason`.
- `src/data_pipeline/build_false_twin_claims.py` builds `claims_false_twins.json` and `claims_paired.json` (150 items: 75 Valid, 75 Invalid).
- The twins are **LLM-drafted and unverified** (`verified_by: llm_unverified`).

**Prompts** (`data/claims/claims_prompts.jsonl`): 13 per item, 1,950 in total.
- One neutral question plus two phrasings for each of six feedback categories.
- Criticism categories: mild doubt, authority, strong rebuttal.
- Praise categories: answer nudge, ownership, flattery.
- Each prompt asks for `[Verdict: Valid|Invalid] [Confidence: X%]` on the first line.
- The build is seeded and reproducible.

```bash
python src/data_pipeline/build_false_twin_claims.py
python src/data_pipeline/build_claims_prompts.py
```

Details: [`docs/claims_dataset.md`](docs/claims_dataset.md).

---

## O1: sustained-pressure trajectory

Each item gets a neutral question (turn 0), then six follow-up turns on one track. There are three tracks:
- **praise**;
- **criticism**;
- **control**, which re-asks the question with no feedback.

After every turn the runner records:
- the parsed verdict (unparseable answers are flagged `parse_fail`, never scored);
- a target-token probe P(Valid) and the derived `truth_confidence`;
- a post-response self-check probe in the style of P(True) (Kadavath et al. 2022);
- the stated confidence.

On JarvisLabs, all three tracks run concurrently on one GPU, followed by the analysis:

```bash
# smoke test: 2 claim pairs
bash scripts/run_jarvislabs_experiment.sh meta-llama/Meta-Llama-3.1-8B-Instruct results/smoke_v3 2
# full run (add a 4th argument "bf16" on a 48 GB+ card; default is 4-bit)
nohup bash scripts/run_jarvislabs_experiment.sh meta-llama/Meta-Llama-3.1-8B-Instruct results/o1_paired_v3 > o1.log 2>&1 &
```

Analysis only:

```bash
python src/evaluation/analyze_o1_paired.py --results_dir results/o1_paired_v3
```

It produces `analysis/summary.md`, CSVs and figures, all broken down by track and truth label:
- accuracy, "Valid" rate and truth confidence per turn, with item-level bootstrap CIs;
- pressure minus control;
- ΔAcc, flip rate and Number of Flips.

**Pilot.** `results/o1_full_run_v2` was produced by an earlier pipeline. That pipeline had rule-generated twins, the five false originals scored as Valid, early stopping, and no control. `src/evaluation/salvage_o1_v2.py` cleans it into `results/o1_v2_salvaged`: the 5 originals are re-scored, only the 17 usable twins are kept, and turns are cut at 2. Treat its results as descriptive only.

---

## O2: reversibility under criticism

```bash
bash scripts/run_jarvislabs_reversibility_experiment.sh meta-llama/Meta-Llama-3.1-8B-Instruct results/reversibility_run
# offline plumbing test (mock model, results are not meaningful):
python src/evaluation/run_reversibility_experiment.py --mock_model --max_turns 6 --limit_claims 10 --output_dir results/o2_mock
```

Known issues before O2 results can be reported:
1. It pools false claims, where criticism is correct, with true ones.
2. Early stopping truncates trajectories.
3. It uses only the pre-response probe.

---

## Tonality scorer

The scorer is a DeBERTa-v3-large encoder with an attention-pooled tanh head, regressing a valence in [−1, 1]. It is trained on about 20k feedback-like prompts in `data/curated`.

On the 1,988-prompt test split (`results/scorer/evaluation_metrics_testset.json`) it scores Pearson r = 0.963, RMSE 0.155, and 88.3% five-bracket accuracy. These numbers overstate its quality:
- Labels are author-assigned with few distinct levels.
- A predictor that only knows the true bracket already reaches r = 0.993.
- Correlation within brackets is low.

Planned validation:
- Best-Worst Scaling on an anchor set;
- CheckList-style behavioural tests.

Model weights are gitignored. The scripts below need `checkpoints/best_deberta_large_curated_scorer/model_weights.pt`.

```bash
bash scripts/run_train_deberta_large.sh
python src/evaluation/evaluate_testset.py
python src/evaluation/sanity_eval.py
python src/data_pipeline/score_claims_prompts.py     # QA of claim prompts against expected brackets
```

---

## Interim report

The ACL-format interim reports are built from these results:
- `src/evaluation/make_report_tables.py` turns the analysis CSVs into LaTeX tables, number macros and figures.
- The report sources live outside this repository.

---

## Tests

```bash
python -m unittest discover tests
```

The O2 tests need `transformers` installed (the module imports it) and use the mock simulator.
