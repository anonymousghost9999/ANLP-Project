# JarvisLabs runbook and plan

## Running on JarvisLabs

1. **Instance.** A PyTorch template with CUDA.
   - A 24 GB card fits three 4-bit Llama-3.1-8B processes.
   - A 48–80 GB card fits three bf16 processes.
2. **Setup** (once per instance):
   ```bash
   git pull
   bash scripts/setup_jarvislabs.sh
   huggingface-cli login            # Llama 3.1 is gated
   python3 scripts/download_model.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct
   ```
3. **Smoke test** (2 claim pairs, about 15–20 min). In each `results/smoke_v3/*/run.log`, check that the "probe token" lines show different tokens for Valid/Invalid and for A/B.
   ```bash
   bash scripts/run_jarvislabs_experiment.sh meta-llama/Meta-Llama-3.1-8B-Instruct results/smoke_v3 2
   ```
4. **Full O1 run.** This launches the praise, criticism and control tracks as three concurrent processes, then runs the analysis and writes a `.tar.gz`.
   ```bash
   nohup bash scripts/run_jarvislabs_experiment.sh meta-llama/Meta-Llama-3.1-8B-Instruct results/o1_paired_v3 > o1.log 2>&1 &
   tail -f results/o1_paired_v3/*/run.log
   ```
   Add `"" bf16` as extra arguments on a 48 GB+ card.
5. **Bring results back.** Download the archive, or commit `results/o1_paired_v3` and push.

**Timing.** The O2 pilot ran at about 8.9 s per turn (4-bit, 128 new tokens, one process). The full O1 run is about 3,150 turns, so expect roughly 3–4 hours of wall time with the three tracks running concurrently.

Each runner streams one JSONL line per turn, so `analyze_o1_paired.py` can analyse a partially finished run.

## Plan to the final submission (31 Oct 2026)

| Week | Work |
| :--- | :--- |
| 3–9 Oct | Full fixed O1 run (praise, criticism, control; 150 items, 6 turns). Two team members verify the 75 false twins. Second model (Qwen2.5-7B-Instruct). |
| 10–16 Oct | O2 on true claims only, without early stopping. O4 confidence floor. Scorer validation: Best–Worst Scaling on an anchor set, CheckList tests. |
| 17–23 Oct | O3: criticism until the first flip, then praise; recovery ratio and recovery turn. O6: k criticism turns then a fixed praise budget, k = 1…6. |
| 24–31 Oct | O5 on a coarse grid of schedules. Calibration over turns (ECE@T). Sampled-decoding robustness check. Final report. |

O5 and O6 are the first to be scoped down if time runs short.

### Planned design changes
- Feedback turns should respond to the model's previous answer instead of re-posing the question.
- The answer-nudge category should name the incorrect verdict explicitly.
- Pressure strength stays categorical until the scorer's valences are validated.
- O2 should be restricted to true claims, run without early stopping, and use the post-response self-check probe.
