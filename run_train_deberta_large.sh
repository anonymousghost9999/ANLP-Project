#!/usr/bin/env bash
# ==============================================================================
# Fine-tune Microsoft DeBERTa-v3-large on the 20k Curated Prompt Dataset
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="./.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

echo "=============================================================================="
echo " Fine-Tuning Microsoft DeBERTa-v3-large on 20k Curated Dataset"
echo "=============================================================================="
echo "  - Backbone Model:       microsoft/deberta-v3-large (435M params)"
echo "  - Dataset Split:        classifier_data/curated (15,936 train, 2,007 val)"
echo "  - Batch Size:           8 per device (Grad Accum 8 = Effective Batch 64)"
echo "  - Optimizer:            Adafactor (Low-memory factored for large models)"
echo "  - Hardware Support:     Auto Mixed Precision (BF16/FP16) + Gradient Checkpointing"
echo "  - Target Artifact:      best_deberta_large_curated_scorer/"
echo "=============================================================================="

$PYTHON_BIN train_scorer.py \
    --model_name "microsoft/deberta-v3-large" \
    --data_dir "classifier_data/curated" \
    --output_dir "best_deberta_large_curated_scorer" \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum_steps 8 \
    --backbone_lr 1e-5 \
    --head_lr 8e-5 \
    --max_length 256 \
    --dropout 0.15 \
    --alpha 0.4 \
    --precision auto \
    --optimizer auto
