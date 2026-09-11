@echo off
echo ==============================================================================
echo  Launching State-of-the-Art DeBERTa-v3 Prompt Scorer Training (RTX 3050 GPU)
echo ==============================================================================
echo  - Backbone:            microsoft/deberta-v3-base
echo  - Per-Device Batch:    8 (Grad Accum 8 = Effective Batch 64)
echo  - Epochs:              3
echo  - Differential LRs:    Backbone: 1.5e-5, Head: 1.0e-4 (Cosine Warmup 10%%)
echo  - Precision:           Hardware Auto Mixed Precision (BF16/FP16 Tensor Cores)
echo  - Multi-Sample Drop:   0.15 (5 sample passes)
echo  - Loss Balancing:      Smooth L1 (beta=0.1) + 0.4 * (1 - Pearson r)
echo ==============================================================================

.\.venv\Scripts\python.exe train_scorer.py ^
    --model_name microsoft/deberta-v3-base ^
    --epochs 3 ^
    --batch_size 8 ^
    --grad_accum_steps 8 ^
    --backbone_lr 1.5e-5 ^
    --head_lr 1.0e-4 ^
    --max_length 256 ^
    --dropout 0.15 ^
    --alpha 0.4 ^
    --precision auto ^
    --output_dir best_deberta_prompt_scorer

pause
