import os
import sys
import time
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress repetitive HF and library warnings for clean console output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from model_architecture import ContinuousPromptScorerModel, HybridPromptScoreLoss

class PromptScoreDataset(Dataset):
    """Dataset for continuous prompt score pairs (text, score)."""
    def __init__(self, texts: list[str], scores: list[float]):
        self.texts = texts
        self.scores = scores

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": str(self.texts[idx]),
            "score": float(self.scores[idx])
        }

def create_collate_fn(tokenizer, max_length=256):
    """Dynamic padding batch collator."""
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        scores = torch.tensor([item["score"] for item in batch], dtype=torch.float32)
        
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded["labels"] = scores
        return encoded
    return collate_fn

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute comprehensive regression & classification bracket metrics."""
    y_pred = np.clip(y_pred, -1.0, 1.0)
    
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    
    # 3-Way Directional Accuracy: Negative (< -0.2), Neutral ([-0.2, 0.2]), Positive (> 0.2)
    def to_ternary(arr):
        res = np.zeros(len(arr))
        res[arr < -0.2] = -1
        res[arr > 0.2] = 1
        return res
    dir_acc = float((to_ternary(y_true) == to_ternary(y_pred)).mean() * 100)
    
    # 5-Way Bracket Accuracy
    def to_5way(arr):
        res = np.zeros(len(arr), dtype=int)
        res[arr <= -0.6] = 0 # Strong Neg
        res[(arr > -0.6) & (arr <= -0.2)] = 1 # Mild Neg
        res[(arr > -0.2) & (arr < 0.2)] = 2  # Neutral
        res[(arr >= 0.2) & (arr < 0.6)] = 3  # Mild Pos
        res[arr >= 0.6] = 4 # Strong Pos
        return res
    bracket_5way_acc = float((to_5way(y_true) == to_5way(y_pred)).mean() * 100)
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": float(pr),
        "spearman": float(sr),
        "directional_accuracy": dir_acc,
        "bracket_5way_accuracy": bracket_5way_acc
    }

def evaluate_model(model, dataloader, device, use_amp=False, amp_dtype=None):
    """Run evaluation on validation or test dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    loss_fn = HybridPromptScoreLoss(alpha=0.5)

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    scores = out["scores"]
                    loss = loss_fn(scores, labels)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                scores = out["scores"]
                loss = loss_fn(scores, labels)

            total_loss += loss.item() * len(labels)
            all_preds.extend(scores.detach().float().cpu().numpy())
            all_labels.extend(labels.detach().float().cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["eval_loss"] = avg_loss
    return metrics, np.array(all_preds)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Continuous Prompt Scorer from Scratch with Bounded Tanh Head and Hybrid Loss.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace model backbone ('sentence-transformers/all-MiniLM-L6-v2', 'microsoft/deberta-v3-small', 'microsoft/deberta-v3-base')")
    parser.add_argument("--data_dir", type=str, default="classifier_data", help="Directory with train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, default="best_prompt_scorer", help="Directory to save best checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (e.g. 32 for MiniLM, 16 for DeBERTa)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps for effective larger batch sizes")
    parser.add_argument("--backbone_lr", type=float, default=2e-5, help="Learning rate for pretrained backbone")
    parser.add_argument("--head_lr", type=float, default=1e-4, help="Learning rate for custom regression head")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token sequence length")
    parser.add_argument("--dropout", type=float, default=0.2, help="Head dropout probability")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of Pearson loss in hybrid loss")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                        help="Compute precision: 'auto' (auto-detects BF16/FP16 on CUDA), 'bf16', 'fp16', or 'fp32'")
    return parser.parse_args()

def setup_hardware_optimizations(device):
    """Enable harmless, high-performance hardware acceleration (TF32, cuDNN benchmark)."""
    if device.type == "cuda":
        # 1. Enable TensorFloat-32 (TF32) on Ampere+ GPUs (RTX 3050, 3080, A100, etc.)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 2. Enable cuDNN autotuner to select the fastest algorithms
        torch.backends.cudnn.benchmark = True
        print("[Compute Optimization] Enabled TF32 precision & cuDNN auto-tuner.", flush=True)

def determine_amp_config(device, requested_precision):
    """Determine whether to use BF16, FP16, or FP32 AMP based on GPU hardware."""
    if device.type != "cuda" or requested_precision == "fp32":
        return False, None, False, "Standard Float32"
    
    if requested_precision == "bf16" or (requested_precision == "auto" and torch.cuda.is_bf16_supported()):
        return True, torch.bfloat16, False, "Mixed Precision (BFloat16 / BF16)"
    
    # Fallback to FP16 with GradScaler
    return True, torch.float16, True, "Mixed Precision (FP16 + GradScaler)"

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_hardware_optimizations(device)
    
    effective_batch = args.batch_size * args.grad_accum_steps
    use_amp, amp_dtype, use_scaler, prec_desc = determine_amp_config(device, args.precision)
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    print("=" * 80, flush=True)
    print(" CONTINUOUS PROMPT SCORER - OPTIMIZED TRAINING PIPELINE", flush=True)
    print("=" * 80, flush=True)
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Backbone Model:   {args.model_name}", flush=True)
    print(f"Epochs:           {args.epochs} | Per-Device Batch: {args.batch_size} | Effective Batch: {effective_batch}", flush=True)
    print(f"Learning Rates:   Backbone: {args.backbone_lr:.1e} | Head: {args.head_lr:.1e}", flush=True)
    print(f"Loss Function:    Hybrid Smooth L1 + {args.alpha} * (1 - Pearson Loss)", flush=True)
    print(f"Precision:        {prec_desc}", flush=True)
    print(f"Output Directory: {args.output_dir}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Data
    train_df = pd.read_json(os.path.join(args.data_dir, "train.jsonl"), lines=True)
    val_df = pd.read_json(os.path.join(args.data_dir, "val.jsonl"), lines=True)
    test_df = pd.read_json(os.path.join(args.data_dir, "test.jsonl"), lines=True)

    print(f"Loaded records -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}", flush=True)

    # 2. Tokenizer & Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = create_collate_fn(tokenizer, max_length=args.max_length)

    train_ds = PromptScoreDataset(train_df["text"].tolist(), train_df["base_score"].tolist())
    val_ds = PromptScoreDataset(val_df["text"].tolist(), val_df["base_score"].tolist())
    test_ds = PromptScoreDataset(test_df["text"].tolist(), test_df["base_score"].tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)

    # 3. Model Architecture
    model = ContinuousPromptScorerModel(args.model_name, dropout_rate=args.dropout).to(device)

    # 4. Differential Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        # Backbone with decay
        {"params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, "lr": args.backbone_lr},
        # Backbone without decay
        {"params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.backbone_lr},
        # Custom Head & Attention Pooling (Higher LR)
        {"params": [p for n, p in model.head.named_parameters()],
         "weight_decay": 0.01, "lr": args.head_lr},
        {"params": [p for n, p in model.attention_weights.named_parameters()],
         "weight_decay": 0.01, "lr": args.head_lr}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    total_opt_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(0.10 * total_opt_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps)
    loss_fn = HybridPromptScoreLoss(alpha=args.alpha)

    # 5. Training Loop
    best_pearson = -1.0
    best_rmse = 999.0
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 105, flush=True)
    print(f" {'Epoch':<7} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Pearson (r)':<16} | {'Val RMSE':<10} | {'Val DirAcc':<11} | {'Time':<8} | {'Status'}", flush=True)
    print("=" * 105, flush=True)

    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        total_batches = len(train_loader)
        
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    scores = out["scores"]
                    raw_loss = loss_fn(scores, labels)
                    loss = raw_loss / args.grad_accum_steps
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                scores = out["scores"]
                raw_loss = loss_fn(scores, labels)
                loss = raw_loss / args.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum_steps == 0 or step == total_batches:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            train_loss += raw_loss.item() * len(labels)
            train_preds.extend(scores.detach().float().cpu().numpy())
            train_labels.extend(labels.detach().float().cpu().numpy())

            # Live in-place progress update every 10 steps
            if step % 10 == 0 or step == total_batches:
                pct = (step / total_batches) * 100
                elapsed = time.time() - epoch_start
                it_per_sec = step / (elapsed + 1e-6)
                sys.stdout.write(f"\r >> [Epoch {epoch}/{args.epochs}] Batch {step:>4}/{total_batches} ({pct:>5.1f}%) | Current Loss: {raw_loss.item():.4f} | Speed: {it_per_sec:.1f} batch/s")
                sys.stdout.flush()

        avg_train_loss = train_loss / len(train_labels)

        # Clear progress line before evaluating
        sys.stdout.write(f"\r >> [Epoch {epoch}/{args.epochs}] Validating model on 2,643 test prompts...{' '*30}\r")
        sys.stdout.flush()

        # Evaluate on Validation set
        val_metrics, _ = evaluate_model(model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
        epoch_sec = time.time() - epoch_start
        status_tag = ""

        # Checkpoint if best Pearson correlation
        if val_metrics["pearson"] > best_pearson:
            best_pearson = val_metrics["pearson"]
            best_rmse = val_metrics["rmse"]
            status_tag = ">>> SAVED (BEST)"
            
            # Save full model weights, tokenizer, and config
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_weights.pt"))
            model.config.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            # Save metadata
            meta = {
                "backbone": args.model_name,
                "best_epoch": epoch,
                "val_pearson": best_pearson,
                "val_rmse": best_rmse,
                "val_metrics": val_metrics
            }
            pd.Series(meta).to_json(os.path.join(args.output_dir, "training_meta.json"))

        # Print formatted row
        sys.stdout.write(f"\r {epoch:^2}/{args.epochs:<3} | {avg_train_loss:<10.4f} | {val_metrics['eval_loss']:<10.4f} | {val_metrics['pearson']:<16.4f} | {val_metrics['rmse']:<10.4f} | {val_metrics['directional_accuracy']:<9.2f}% | {epoch_sec:>5.1f}s   | {status_tag}\n")
        sys.stdout.flush()

    print("=" * 105, flush=True)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes (Best Val Pearson r: {best_pearson:.4f}, Best RMSE: {best_rmse:.4f})", flush=True)

    # 6. Final Evaluation on Test Set using Best Checkpoint
    print("\n" + "=" * 80, flush=True)
    print(" FINAL EVALUATION ON UNSEEN TEST SET (BEST CHECKPOINT)", flush=True)
    print("=" * 80, flush=True)
    
    # Reload best weights
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_weights.pt"), map_location=device))
    test_metrics, test_preds = evaluate_model(model, test_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)

    print(f"  Pearson Correlation (r):  {test_metrics['pearson']:.4f} ({test_metrics['pearson']*100:.2f}%)", flush=True)
    print(f"  Spearman Correlation (\u03c1): {test_metrics['spearman']:.4f} ({test_metrics['spearman']*100:.2f}%)", flush=True)
    print(f"  Root Mean Squared Error:  {test_metrics['rmse']:.4f}", flush=True)
    print(f"  Mean Absolute Error:      {test_metrics['mae']:.4f}", flush=True)
    print(f"  R\u00b2 Goodness-of-Fit:       {test_metrics['r2']:.4f}", flush=True)
    print(f"  3-Way Directional Acc:    {test_metrics['directional_accuracy']:.2f}%", flush=True)
    print(f"  5-Way Semantic Bracket:   {test_metrics['bracket_5way_accuracy']:.2f}%", flush=True)
    print("=" * 80, flush=True)
    print(f"Best model artifact saved at: {os.path.abspath(args.output_dir)}", flush=True)

if __name__ == "__main__":
    main()
