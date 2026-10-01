#!/usr/bin/env bash
# ==============================================================================
# End-to-End Production Experiment Execution Script for JarvisLabs
# ==============================================================================

set -e

MODEL_NAME="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${2:-results/production_run}"
LIMIT_CLAIMS="${3:-}"
TRACK="${4:-positive}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo " STARTING PRODUCTION EXPERIMENT RUN ON JARVISLABS GPU"
echo "=============================================================================="
echo " Model Target:     $MODEL_NAME"
echo " Output Directory: $OUTPUT_DIR"
echo " Track Selection:  $TRACK"
if [ -n "$LIMIT_CLAIMS" ]; then
    echo " Claim Limit:      First $LIMIT_CLAIMS claims (Subsetting)"
    LIMIT_FLAG="--limit_claims $LIMIT_CLAIMS"
else
    echo " Claim Limit:      All 75 claims"
    LIMIT_FLAG=""
fi
echo "=============================================================================="

# 1. Run Trajectory Experiment (Positive Track)
echo -e "\n>>> [Phase 1/3] Running Multi-Turn Trajectory Experiment ($TRACK track)..."
python3 src/evaluation/run_claims_experiment.py \
    --model_name "$MODEL_NAME" \
    --prompts_path "data/claims/claims_prompts.jsonl" \
    --claims_json_path "data/claims/claims.json" \
    --output_dir "$OUTPUT_DIR" \
    --track "$TRACK" \
    --confidence_floor 50.0 \
    --settled_turns_k 2 \
    --stalled_turns_m 3 \
    $LIMIT_FLAG

# 2. Run Figure Generation & Mixed-Effects Statistical Analysis
echo -e "\n>>> [Phase 2/3] Generating Trajectory Figures & Statistical Analysis..."
python3 src/evaluation/analyze_results.py \
    --results_csv "$OUTPUT_DIR/claims_drift_summary.csv" \
    --output_dir "$OUTPUT_DIR/figures"

# 3. Compress Results for Fast Archive Download
echo -e "\n>>> [Phase 3/3] Archiving Results for Easy Download..."
ARCHIVE_NAME="jarvislabs_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czvf "$ARCHIVE_NAME" "$OUTPUT_DIR"

echo "=============================================================================="
echo " EXPERIMENT COMPLETED SUCCESSFULLY!"
echo " Results Archive: $ARCHIVE_NAME"
echo " Tabular CSV:     $OUTPUT_DIR/claims_drift_summary.csv"
echo " Trajectory JSON: $OUTPUT_DIR/claims_drift_results.jsonl"
echo " Figures Dir:     $OUTPUT_DIR/figures/"
echo "=============================================================================="
