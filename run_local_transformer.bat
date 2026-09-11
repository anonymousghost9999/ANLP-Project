@echo off
echo ==============================================================================
echo  Launching Optimized Continuous Prompt Scorer Training (RTX 3050 GPU)
echo ==============================================================================
echo  - Backbone:            sentence-transformers/all-MiniLM-L6-v2
echo  - Per-Device Batch:    32 (Grad Accum 2 = Effective Batch 64)
echo  - Epochs:              4
echo  - Differential LRs:    Backbone: 2.5e-5, Head: 1.5e-4 (Cosine Warmup 10%%)
echo  - Precision:           AMP FP16 + TF32 (Ampere Tensor Cores)
echo  - Multi-Sample Drop:   0.15 (5 sample passes)
echo  - Loss Balancing:      Smooth L1 (beta=0.1) + 0.4 * (1 - Pearson r)
echo ==============================================================================

.\.venv\Scripts\python.exe train_scorer.py ^
    --model_name sentence-transformers/all-MiniLM-L6-v2 ^
    --epochs 4 ^
    --batch_size 32 ^
    --grad_accum_steps 2 ^
    --backbone_lr 2.5e-5 ^
    --head_lr 1.5e-4 ^
    --max_length 256 ^
    --dropout 0.15 ^
    --alpha 0.4 ^
    --output_dir best_prompt_scorer

pause
