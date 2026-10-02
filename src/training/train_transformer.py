import os
# Configure PyTorch memory allocator to avoid CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress repetitive warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor

# Safe import from models
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.transformer import ContinuousPromptScorerModel, HybridPromptScoreLoss

DEFAULT_CURATED_DATA_DIR = os.path.join(REPO_ROOT, "data", "curated")
DEFAULT_DATA_DIR = DEFAULT_CURATED_DATA_DIR if os.path.exists(DEFAULT_CURATED_DATA_DIR) else os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "best_deberta_large_curated_scorer")


class PromptScoreDataset(Dataset):
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
    y_pred = np.clip(y_pred, -1.0, 1.0)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    
    def to_ternary(arr):
        res = np.zeros(len(arr))
        res[arr < -0.2] = -1
        res[arr > 0.2] = 1
        return res
    dir_acc = float((to_ternary(y_true) == to_ternary(y_pred)).mean() * 100)
    
    def to_5way(arr):
        res = np.zeros(len(arr), dtype=int)
        res[arr <= -0.6] = 0
        res[(arr > -0.6) & (arr <= -0.2)] = 1
        res[(arr > -0.2) & (arr < 0.2)] = 2
        res[(arr >= 0.2) & (arr < 0.6)] = 3
        res[arr >= 0.6] = 4
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
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large",
                        help="HuggingFace backbone ('microsoft/deberta-v3-large', 'microsoft/deberta-v3-base', 'sentence-transformers/all-MiniLM-L6-v2')")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save best checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (e.g. 8 for DeBERTa on 11GB VRAM, 16/32 for base/MiniLM)")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--backbone_lr", type=float, default=1.0e-5, help="Learning rate for pretrained backbone (e.g. 1e-5 for large, 1.5e-5 for base)")
    parser.add_argument("--head_lr", type=float, default=8.0e-5, help="Learning rate for custom regression head")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum token sequence length")
    parser.add_argument("--dropout", type=float, default=0.15, help="Head dropout probability")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of Pearson loss in hybrid loss")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["auto", "adafactor", "adamw"],
                        help="Optimizer: 'auto' (adafactor for large models, adamw for others), 'adafactor', or 'adamw'")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save VRAM on large transformer backbones")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                        help="Compute precision: 'auto', 'bf16', 'fp16', or 'fp32'")
    return parser.parse_args()


def setup_hardware_optimizations(device):
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("[Compute Optimization] Enabled TF32 precision & cuDNN auto-tuner.", flush=True)


def determine_amp_config(device, requested_precision):
    if device.type != "cuda" or requested_precision == "fp32":
        return False, None, False, "Standard Float32"
    if requested_precision == "bf16" or (requested_precision == "auto" and torch.cuda.is_bf16_supported()):
        return True, torch.bfloat16, False, "Mixed Precision (BFloat16 / BF16)"
    return True, torch.float16, True, "Mixed Precision (FP16 + GradScaler)"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_hardware_optimizations(device)
    
    effective_batch = args.batch_size * args.grad_accum_steps
    use_amp, amp_dtype, use_scaler, prec_desc = determine_amp_config(device, args.precision)
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    # Determine optimizer type
    opt_choice = args.optimizer
    if opt_choice == "auto":
        opt_choice = "adafactor" if "large" in args.model_name.lower() else "adamw"

    print("=" * 80, flush=True)
    print(" CONTINUOUS PROMPT SCORER - OPTIMIZED TRAINING PIPELINE", flush=True)
    print("=" * 80, flush=True)
    print(f"Device:           {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Backbone Model:   {args.model_name}", flush=True)
    print(f"Epochs:           {args.epochs} | Per-Device Batch: {args.batch_size} | Effective Batch: {effective_batch}", flush=True)
    print(f"Learning Rates:   Backbone: {args.backbone_lr:.1e} | Head: {args.head_lr:.1e}", flush=True)
    print(f"Loss Function:    Hybrid Smooth L1 + {args.alpha} * (1 - Pearson Loss)", flush=True)
    print(f"Precision:        {prec_desc}", flush=True)
    print(f"Optimizer:        {opt_choice.upper()} {'(low-memory factored)' if opt_choice == 'adafactor' else '(standard)'}", flush=True)
    print(f"Grad Checkpoint:  {'Enabled' if args.gradient_checkpointing else 'Disabled'}", flush=True)
    print(f"Output Directory: {args.output_dir}", flush=True)
    print("=" * 80, flush=True)

    train_df = pd.read_json(os.path.join(args.data_dir, "train.jsonl"), lines=True)
    val_df = pd.read_json(os.path.join(args.data_dir, "val.jsonl"), lines=True)
    test_df = pd.read_json(os.path.join(args.data_dir, "test.jsonl"), lines=True)

    print(f"Loaded records -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = create_collate_fn(tokenizer, max_length=args.max_length)

    train_ds = PromptScoreDataset(train_df["text"].tolist(), train_df["base_score"].tolist())
    val_ds = PromptScoreDataset(val_df["text"].tolist(), val_df["base_score"].tolist())
    test_ds = PromptScoreDataset(test_df["text"].tolist(), test_df["base_score"].tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, collate_fn=collate_fn)

    model = ContinuousPromptScorerModel(args.model_name, dropout_rate=args.dropout).to(device)

    # Enable Gradient Checkpointing if requested
    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        print("[Memory Optimization] Enabled Gradient Checkpointing on model backbone.", flush=True)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, "lr": args.backbone_lr},
        {"params": [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": args.backbone_lr},
        {"params": [p for n, p in model.head.named_parameters()],
         "weight_decay": 0.01, "lr": args.head_lr},
        {"params": [p for n, p in model.attention_weights.named_parameters()],
         "weight_decay": 0.01, "lr": args.head_lr}
    ]

    if opt_choice == "adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=args.backbone_lr
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_opt_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(0.10 * total_opt_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps)
    loss_fn = HybridPromptScoreLoss(alpha=args.alpha)

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

            if step % 10 == 0 or step == total_batches:
                pct = (step / total_batches) * 100
                elapsed = time.time() - epoch_start
                it_per_sec = step / (elapsed + 1e-6)
                sys.stdout.write(f"\r >> [Epoch {epoch}/{args.epochs}] Batch {step:>4}/{total_batches} ({pct:>5.1f}%) | Current Loss: {raw_loss.item():.4f} | Speed: {it_per_sec:.1f} batch/s")
                sys.stdout.flush()

        avg_train_loss = train_loss / len(train_labels)
        sys.stdout.write(f"\r >> [Epoch {epoch}/{args.epochs}] Validating model...{' '*40}\r")
        sys.stdout.flush()

        val_metrics, _ = evaluate_model(model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
        epoch_sec = time.time() - epoch_start
        status_tag = ""

        if val_metrics["pearson"] > best_pearson:
            best_pearson = val_metrics["pearson"]
            best_rmse = val_metrics["rmse"]
            status_tag = ">>> SAVED (BEST)"
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_weights.pt"))
            model.config.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            meta = {
                "backbone": args.model_name,
                "best_epoch": epoch,
                "val_pearson": best_pearson,
                "val_rmse": best_rmse,
                "val_metrics": val_metrics
            }
            pd.Series(meta).to_json(os.path.join(args.output_dir, "training_meta.json"))

        sys.stdout.write(f"\r {epoch:^2}/{args.epochs:<3} | {avg_train_loss:<10.4f} | {val_metrics['eval_loss']:<10.4f} | {val_metrics['pearson']:<16.4f} | {val_metrics['rmse']:<10.4f} | {val_metrics['directional_accuracy']:<9.2f}% | {epoch_sec:>5.1f}s   | {status_tag}\n")
        sys.stdout.flush()

    print("=" * 105, flush=True)
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes (Best Val Pearson r: {best_pearson:.4f}, Best RMSE: {best_rmse:.4f})", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(" FINAL EVALUATION ON UNSEEN TEST SET (BEST CHECKPOINT)", flush=True)
    print("=" * 80, flush=True)
    
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "model_weights.pt"), map_location=device))
    test_metrics, test_preds = evaluate_model(model, test_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)

    print(f"  Pearson Correlation (r):  {test_metrics['pearson']:.4f} ({test_metrics['pearson']*100:.2f}%)", flush=True)
    print(f"  Spearman Correlation (ρ): {test_metrics['spearman']:.4f} ({test_metrics['spearman']*100:.2f}%)", flush=True)
    print(f"  Root Mean Squared Error:  {test_metrics['rmse']:.4f}", flush=True)
    print(f"  Mean Absolute Error:      {test_metrics['mae']:.4f}", flush=True)
    print(f"  R² Goodness-of-Fit:       {test_metrics['r2']:.4f}", flush=True)
    print(f"  3-Way Directional Acc:    {test_metrics['directional_accuracy']:.2f}%", flush=True)
    print(f"  5-Way Semantic Bracket:   {test_metrics['bracket_5way_accuracy']:.2f}%", flush=True)
    print("=" * 80, flush=True)
    print(f"Best model artifact saved at: {os.path.abspath(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()
