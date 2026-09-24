import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "fine_tuned_prompt_scorer")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformer regressor for continuous prompt scoring [-1.0, 1.0] using HuggingFace Trainer.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Base HuggingFace model (e.g. 'sentence-transformers/all-MiniLM-L6-v2', 'microsoft/deberta-v3-base')")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing train.jsonl, val.jsonl, test.jsonl")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save fine-tuned model")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache HuggingFace models")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence token length")
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(), help="Use FP16 mixed precision if GPU available")
    return parser.parse_args()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.squeeze(predictions)
    preds = np.clip(preds, -1.0, 1.0)
    
    mse = mean_squared_error(labels, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(labels, preds))
    r2 = float(r2_score(labels, preds))
    pr, _ = pearsonr(labels, preds)
    sr, _ = spearmanr(labels, preds)
    
    def to_ternary(arr):
        res = np.zeros(len(arr))
        res[arr < -0.2] = -1
        res[arr > 0.2] = 1
        return res
        
    bracket_acc = float((to_ternary(labels) == to_ternary(preds)).mean() * 100)
    
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": float(pr),
        "spearman": float(sr),
        "directional_accuracy": bracket_acc
    }


def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device Count: {gpu_count} | Primary GPU: {gpu_name}", flush=True)
        print(f"CUDA Version: {torch.version.cuda}", flush=True)
    else:
        print("Running on CPU mode (No GPU detected).", flush=True)
        
    print(f"Loading base model: {args.model_name}", flush=True)
    if args.cache_dir:
        print(f"Using HuggingFace cache directory: {args.cache_dir}", flush=True)
        os.makedirs(args.cache_dir, exist_ok=True)

    train_path = os.path.join(args.data_dir, "train.jsonl")
    val_path = os.path.join(args.data_dir, "val.jsonl")
    test_path = os.path.join(args.data_dir, "test.jsonl")

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    print(f"Loaded records -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression",
        cache_dir=args.cache_dir
    )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "base_score"]].rename(columns={"base_score": "label"}))
    val_ds = Dataset.from_pandas(val_df[["text", "base_score"]].rename(columns={"base_score": "label"}))
    test_ds = Dataset.from_pandas(test_df[["text", "base_score"]].rename(columns={"base_score": "label"}))

    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    test_tokenized = test_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print("\nStarting fine-tuning...", flush=True)
    trainer.train()

    print("\nEvaluating on Test Set...", flush=True)
    test_results = trainer.evaluate(eval_dataset=test_tokenized)
    print("\n=== Final Test Results ===", flush=True)
    for k, v in test_results.items():
        print(f"  {k}: {v}", flush=True)

    print(f"\nSaving best fine-tuned model and tokenizer to: {args.output_dir}", flush=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully!", flush=True)


if __name__ == "__main__":
    main()
