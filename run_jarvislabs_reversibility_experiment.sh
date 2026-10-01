#!/usr/bin/env bash
# ==============================================================================
# End-to-End Reversibility Under Criticism Experiment Launcher for JarvisLabs
# ==============================================================================

set -e

MODEL_NAME="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${2:-results/reversibility_production_run}"
LIMIT_CLAIMS="${3:-}"
MAX_TURNS="${4:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo " STARTING REVERSIBILITY & OSCILLATION EXPERIMENT ON JARVISLABS GPU"
echo "=============================================================================="
echo " Model Target:     $MODEL_NAME"
echo " Output Directory: $OUTPUT_DIR"
echo " Max Turns:        $MAX_TURNS"
if [ -n "$LIMIT_CLAIMS" ]; then
    echo " Claim Limit:      First $LIMIT_CLAIMS claims (Subsetting)"
    LIMIT_FLAG="--limit_claims $LIMIT_CLAIMS"
else
    echo " Claim Limit:      All 75 claims"
    LIMIT_FLAG=""
fi
echo "=============================================================================="

# 1. Run Complete Reversibility Experiment
echo -e "\n>>> [Phase 1/3] Running Reversibility & Oscillation Experiment..."
python3 src/evaluation/run_reversibility_experiment.py \
    --model_name "$MODEL_NAME" \
    --prompts_path "data/claims/claims_prompts.jsonl" \
    --claims_json_path "data/claims/claims_paired.json" \
    --output_dir "$OUTPUT_DIR" \
    --max_turns "$MAX_TURNS" \
    --load_in_4bit \
    --max_new_tokens 128 \
    --confidence_floor 50.0 \
    --settled_k 2 \
    --stalled_m 3 \
    --order "escalating" \
    $LIMIT_FLAG

# 2. Run Comprehensive Statistical Analysis & Visualizations
echo -e "\n>>> [Phase 2/3] Generating Reversibility Visualizations & Statistical Tests..."
python3 src/evaluation/analyze_reversibility.py \
    --summary_csv "$OUTPUT_DIR/reversibility_summary.csv" \
    --claim_summary_csv "$OUTPUT_DIR/reversibility_claim_summary.csv" \
    --output_dir "$OUTPUT_DIR/figures"

# 3. Compress Results for Fast Archive Download
echo -e "\n>>> [Phase 3/3] Archiving Reversibility Results for Download..."
ARCHIVE_NAME="jarvislabs_reversibility_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czvf "$ARCHIVE_NAME" "$OUTPUT_DIR"

echo "=============================================================================="
echo " REVERSIBILITY EXPERIMENT COMPLETED SUCCESSFULLY!"
echo " Results Archive:   $ARCHIVE_NAME"
echo " Turn Summary CSV:  $OUTPUT_DIR/reversibility_summary.csv"
echo " Claim Summary CSV: $OUTPUT_DIR/reversibility_claim_summary.csv"
echo " Metrics Report:    $OUTPUT_DIR/reversibility_metrics_report.json"
echo " Figures Dir:       $OUTPUT_DIR/figures/"
echo "=============================================================================="

