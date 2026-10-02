#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Continuous Prompt Scorer (DeBERTa-v3 Large Curated Model)
Evaluates on the curated test set (data/curated/test.jsonl) across all regression,
ranking, discretization, tolerance, and subgroup metrics.
"""

import os
import sys
import time
import json
import argparse
import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    max_error,
    r2_score,
    explained_variance_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

# Suppress noisy warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.engine import PromptScorer, DEFAULT_MODEL_DIR


def to_ternary(arr: np.ndarray) -> np.ndarray:
    """Discretize continuous score into 3 categories: Negative (-1), Neutral (0), Positive (+1)."""
    res = np.zeros(len(arr), dtype=int)
    res[arr < -0.2] = -1
    res[arr > 0.2] = 1
    return res


def to_5way_brackets(arr: np.ndarray) -> np.ndarray:
    """
    Discretize continuous score into 5 brackets:
      0: Strongly Negative (<= -0.60)
      1: Mild Negative (-0.60 < s <= -0.20)
      2: Neutral (-0.20 < s < 0.20)
      3: Mild Positive (0.20 <= s < 0.60)
      4: Strongly Positive (>= 0.60)
    """
    res = np.zeros(len(arr), dtype=int)
    res[arr <= -0.60] = 0
    res[(arr > -0.60) & (arr <= -0.20)] = 1
    res[(arr > -0.20) & (arr < 0.20)] = 2
    res[(arr >= 0.20) & (arr < 0.60)] = 3
    res[arr >= 0.60] = 4
    return res


BRACKET_NAMES_5WAY = [
    "Strong Negative (<= -0.60)",
    "Mild Negative (-0.60 to -0.20)",
    "Neutral (-0.20 to +0.20)",
    "Mild Positive (+0.20 to +0.60)",
    "Strong Positive (>= +0.60)",
]

TERNARY_NAMES = [
    "Negative (< -0.20)",
    "Neutral ([-0.20, +0.20])",
    "Positive (> +0.20)",
]


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strata: List[str] = None,
    sources: List[str] = None,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics."""
    # 1. Regression Metrics
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    max_err = float(max_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))

    abs_errors = np.abs(y_pred - y_true)
    percentiles = {
        "p25": float(np.percentile(abs_errors, 25)),
        "p50": float(np.percentile(abs_errors, 50)),
        "p75": float(np.percentile(abs_errors, 75)),
        "p90": float(np.percentile(abs_errors, 90)),
        "p95": float(np.percentile(abs_errors, 95)),
        "p99": float(np.percentile(abs_errors, 99)),
    }

    # 2. Correlation & Ranking Metrics
    pr, p_val_pr = pearsonr(y_true, y_pred)
    sr, p_val_sr = spearmanr(y_true, y_pred)
    try:
        tau, p_val_tau = kendalltau(y_true, y_pred)
    except Exception:
        tau, p_val_tau = float("nan"), float("nan")

    # 3. Tolerance Band Accuracies
    tol_005 = float(np.mean(abs_errors <= 0.05) * 100)
    tol_010 = float(np.mean(abs_errors <= 0.10) * 100)
    tol_015 = float(np.mean(abs_errors <= 0.15) * 100)
    tol_020 = float(np.mean(abs_errors <= 0.20) * 100)
    tol_025 = float(np.mean(abs_errors <= 0.25) * 100)

    # 4. 3-Way Directional Discretization
    y_true_tern = to_ternary(y_true)
    y_pred_tern = to_ternary(y_pred)
    dir_acc = float(accuracy_score(y_true_tern, y_pred_tern) * 100)
    dir_macro_f1 = float(f1_score(y_true_tern, y_pred_tern, average="macro") * 100)
    dir_weighted_f1 = float(f1_score(y_true_tern, y_pred_tern, average="weighted") * 100)
    dir_cm = confusion_matrix(y_true_tern, y_pred_tern, labels=[-1, 0, 1]).tolist()

    # Per-class 3-way metrics
    dir_prec = precision_score(y_true_tern, y_pred_tern, labels=[-1, 0, 1], average=None, zero_division=0)
    dir_rec = recall_score(y_true_tern, y_pred_tern, labels=[-1, 0, 1], average=None, zero_division=0)
    dir_f1 = f1_score(y_true_tern, y_pred_tern, labels=[-1, 0, 1], average=None, zero_division=0)
    ternary_per_class = {}
    for idx, name in enumerate(TERNARY_NAMES):
        ternary_per_class[name] = {
            "precision": float(dir_prec[idx] * 100),
            "recall": float(dir_rec[idx] * 100),
            "f1": float(dir_f1[idx] * 100),
            "support": int(np.sum(y_true_tern == [-1, 0, 1][idx])),
        }

    # 5. 5-Way Bracket Discretization
    y_true_5way = to_5way_brackets(y_true)
    y_pred_5way = to_5way_brackets(y_pred)
    b5_acc = float(accuracy_score(y_true_5way, y_pred_5way) * 100)
    b5_macro_f1 = float(f1_score(y_true_5way, y_pred_5way, average="macro") * 100)
    b5_weighted_f1 = float(f1_score(y_true_5way, y_pred_5way, average="weighted") * 100)
    b5_cm = confusion_matrix(y_true_5way, y_pred_5way, labels=[0, 1, 2, 3, 4]).tolist()

    # Per-class 5-way metrics
    b5_prec = precision_score(y_true_5way, y_pred_5way, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    b5_rec = recall_score(y_true_5way, y_pred_5way, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    b5_f1 = f1_score(y_true_5way, y_pred_5way, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    bracket_per_class = {}
    for idx, name in enumerate(BRACKET_NAMES_5WAY):
        bracket_per_class[name] = {
            "precision": float(b5_prec[idx] * 100),
            "recall": float(b5_rec[idx] * 100),
            "f1": float(b5_f1[idx] * 100),
            "support": int(np.sum(y_true_5way == idx)),
        }

    # 6. Binary Non-Neutral Polarity (Pos vs Neg for |score| >= 0.2)
    non_neutral_mask = np.abs(y_true) >= 0.20
    if np.sum(non_neutral_mask) > 0:
        y_true_bin = (y_true[non_neutral_mask] > 0).astype(int)
        y_pred_bin = (y_pred[non_neutral_mask] > 0).astype(int)
        bin_acc = float(accuracy_score(y_true_bin, y_pred_bin) * 100)
        bin_f1 = float(f1_score(y_true_bin, y_pred_bin) * 100)
    else:
        bin_acc, bin_f1 = float("nan"), float("nan")

    # 7. Subgroup Metrics by Strata
    strata_metrics = {}
    if strata is not None:
        strata_arr = np.array(strata)
        for s in np.unique(strata_arr):
            mask = strata_arr == s
            yt_s, yp_s = y_true[mask], y_pred[mask]
            if len(yt_s) > 1:
                s_pr, _ = pearsonr(yt_s, yp_s) if len(np.unique(yt_s)) > 1 and len(np.unique(yp_s)) > 1 else (0.0, 0.0)
            else:
                s_pr = float("nan")
            strata_metrics[str(s)] = {
                "count": int(np.sum(mask)),
                "mae": float(mean_absolute_error(yt_s, yp_s)),
                "rmse": float(np.sqrt(mean_squared_error(yt_s, yp_s))),
                "pearson": float(s_pr),
                "bracket_acc": float((to_5way_brackets(yt_s) == to_5way_brackets(yp_s)).mean() * 100),
                "dir_acc": float((to_ternary(yt_s) == to_ternary(yp_s)).mean() * 100),
            }

    # 8. Subgroup Metrics by Top Sources
    source_metrics = {}
    if sources is not None:
        src_arr = np.array(sources)
        top_sources = pd.Series(src_arr).value_counts().head(8).index.tolist()
        for src in top_sources:
            mask = src_arr == src
            yt_src, yp_src = y_true[mask], y_pred[mask]
            source_metrics[str(src)] = {
                "count": int(np.sum(mask)),
                "mae": float(mean_absolute_error(yt_src, yp_src)),
                "rmse": float(np.sqrt(mean_squared_error(yt_src, yp_src))),
                "dir_acc": float((to_ternary(yt_src) == to_ternary(yp_src)).mean() * 100),
            }

    return {
        "sample_count": len(y_true),
        "regression": {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "median_absolute_error": medae,
            "max_error": max_err,
            "r2_score": r2,
            "explained_variance": evs,
            "error_percentiles": percentiles,
        },
        "correlation": {
            "pearson": float(pr),
            "pearson_p_value": float(p_val_pr),
            "spearman": float(sr),
            "spearman_p_value": float(p_val_sr),
            "kendall_tau": float(tau),
            "kendall_p_value": float(p_val_tau),
        },
        "tolerances": {
            "within_0.05": tol_005,
            "within_0.10": tol_010,
            "within_0.15": tol_015,
            "within_0.20": tol_020,
            "within_0.25": tol_025,
        },
        "directional_3way": {
            "accuracy": dir_acc,
            "macro_f1": dir_macro_f1,
            "weighted_f1": dir_weighted_f1,
            "per_class": ternary_per_class,
            "confusion_matrix": dir_cm,
        },
        "bracket_5way": {
            "accuracy": b5_acc,
            "macro_f1": b5_macro_f1,
            "weighted_f1": b5_weighted_f1,
            "per_class": bracket_per_class,
            "confusion_matrix": b5_cm,
        },
        "binary_polarity": {
            "support": int(np.sum(non_neutral_mask)),
            "accuracy": bin_acc,
            "f1_score": bin_f1,
        },
        "strata_breakdown": strata_metrics,
        "source_breakdown": source_metrics,
    }


def print_evaluation_report(results: Dict[str, Any], timing_info: Dict[str, Any], model_name: str, test_file: str):
    """Print an aesthetic, structured evaluation report to stdout."""
    reg = results["regression"]
    corr = results["correlation"]
    tols = results["tolerances"]
    d3 = results["directional_3way"]
    b5 = results["bracket_5way"]
    bp = results["binary_polarity"]

    print("\n" + "=" * 92)
    print(f" CONTINUOUS PROMPT SCORER - PERFORMANCE BENCHMARK REPORT")
    print(f" Model:     {model_name}")
    print(f" Test Set:  {test_file} (N = {results['sample_count']:,} samples)")
    print(f" Hardware:  {timing_info['device']} | Batch Size: {timing_info['batch_size']}")
    print(f" Speed:     {timing_info['throughput']:.1f} samples/sec ({timing_info['latency_ms']:.2f} ms/sample)")
    print("=" * 92)

    # Summary Card
    print("\n" + "-" * 92)
    print(" [1] EXECUTIVE SUMMARY METRICS")
    print("-" * 92)
    print(f"  Pearson Correlation (r):          {corr['pearson']:>8.4f}   (p-value: {corr['pearson_p_value']:.2e})")
    print(f"  Spearman Rank Correlation (\u03c1):     {corr['spearman']:>8.4f}   (p-value: {corr['spearman_p_value']:.2e})")
    print(f"  Kendall's Tau (\u03c4):                 {corr['kendall_tau']:>8.4f}   (p-value: {corr['kendall_p_value']:.2e})")
    print(f"  R\u00b2 Score (Variance Explained):     {reg['r2_score']:>8.4f}")
    print(f"  Root Mean Squared Error (RMSE):   {reg['rmse']:>8.4f}")
    print(f"  Mean Absolute Error (MAE):        {reg['mae']:>8.4f}")
    print(f"  Median Absolute Error (MedAE):    {reg['median_absolute_error']:>8.4f}")
    print(f"  3-Way Directional Accuracy:       {d3['accuracy']:>7.2f}%  (Macro F1: {d3['macro_f1']:.2f}%)")
    print(f"  5-Way Bracket Accuracy:           {b5['accuracy']:>7.2f}%  (Macro F1: {b5['macro_f1']:.2f}%)")
    print(f"  Binary Polarity Accuracy (Pos/Neg):{bp['accuracy']:>6.2f}%  (F1: {bp['f1_score']:.2f}%, N={bp['support']:,})")

    # Error Tolerance Bands
    print("\n" + "-" * 92)
    print(" [2] ERROR TOLERANCE RESIDUAL ANALYSIS")
    print("-" * 92)
    print(f"  Predictions within \u00b10.05 of true score:  {tols['within_0.05']:>6.2f}%")
    print(f"  Predictions within \u00b10.10 of true score:  {tols['within_0.10']:>6.2f}%")
    print(f"  Predictions within \u00b10.15 of true score:  {tols['within_0.15']:>6.2f}%")
    print(f"  Predictions within \u00b10.20 of true score:  {tols['within_0.20']:>6.2f}%")
    print(f"  Predictions within \u00b10.25 of true score:  {tols['within_0.25']:>6.2f}%")
    p = reg["error_percentiles"]
    print(f"  Residual Percentiles -> 25th: {p['p25']:.4f} | 50th: {p['p50']:.4f} | 75th: {p['p75']:.4f} | 90th: {p['p90']:.4f} | 95th: {p['p95']:.4f}")

    # 3-Way Directional Breakdown
    print("\n" + "-" * 92)
    print(" [3] 3-WAY DIRECTIONAL CLASSIFICATION BREAKDOWN")
    print("-" * 92)
    print(f"  {'Class Label':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("  " + "-" * 88)
    for c_name, c_dict in d3["per_class"].items():
        print(f"  {c_name:<28} | {c_dict['precision']:>8.2f}%  | {c_dict['recall']:>8.2f}%  | {c_dict['f1']:>8.2f}%  | {c_dict['support']:>7,}")
    print("  " + "-" * 88)
    print(f"  {'Macro Average':<28} | {'-':>10} | {'-':>10} | {d3['macro_f1']:>8.2f}%  | {results['sample_count']:>7,}")
    print(f"  {'Weighted Average':<28} | {'-':>10} | {'-':>10} | {d3['weighted_f1']:>8.2f}%  | {results['sample_count']:>7,}")

    # 5-Way Bracket Breakdown
    print("\n" + "-" * 92)
    print(" [4] 5-WAY BRACKET CLASSIFICATION BREAKDOWN")
    print("-" * 92)
    print(f"  {'Bracket':<32} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("  " + "-" * 88)
    for b_name, b_dict in b5["per_class"].items():
        print(f"  {b_name:<32} | {b_dict['precision']:>8.2f}%  | {b_dict['recall']:>8.2f}%  | {b_dict['f1']:>8.2f}%  | {b_dict['support']:>7,}")
    print("  " + "-" * 88)
    print(f"  {'Macro Average':<32} | {'-':>10} | {'-':>10} | {b5['macro_f1']:>8.2f}%  | {results['sample_count']:>7,}")
    print(f"  {'Weighted Average':<32} | {'-':>10} | {'-':>10} | {b5['weighted_f1']:>8.2f}%  | {results['sample_count']:>7,}")

    # Subgroup Breakdown by Strata
    if results.get("strata_breakdown"):
        print("\n" + "-" * 92)
        print(" [5] STRATA SUBGROUP PERFORMANCE")
        print("-" * 92)
        print(f"  {'Strata':<18} | {'Count':<7} | {'MAE':<8} | {'RMSE':<8} | {'Bracket Acc':<12} | {'Dir Acc':<10}")
        print("  " + "-" * 88)
        for s_name, s_dict in results["strata_breakdown"].items():
            print(f"  {s_name:<18} | {s_dict['count']:>6,} | {s_dict['mae']:>7.4f}  | {s_dict['rmse']:>7.4f}  | {s_dict['bracket_acc']:>10.2f}%  | {s_dict['dir_acc']:>8.2f}%")

    print("\n" + "=" * 92)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Continuous DeBERTa Scorer on Curated Test Set.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to trained model directory (defaults to auto-detected best checkpoint)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "curated", "test.jsonl"),
        help="Path to test.jsonl file",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu', or auto)")
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join(REPO_ROOT, "results", "scorer", "evaluation_metrics_testset.json"),
        help="Path to save output metrics JSON",
    )
    parser.add_argument(
        "--save_predictions",
        type=str,
        default=None,
        help="Optional path to save per-sample predictions jsonl",
    )
    args = parser.parse_args()

    # Determine model directory
    model_dir = args.model_dir
    if model_dir is None:
        if os.path.isfile(os.path.join(REPO_ROOT, "checkpoints", "best_deberta_large_curated_scorer", "model_weights.pt")):
            model_dir = os.path.join(REPO_ROOT, "checkpoints", "best_deberta_large_curated_scorer")
        elif os.path.isfile(os.path.join(REPO_ROOT, "best_deberta_large_curated_scorer", "model_weights.pt")):
            model_dir = os.path.join(REPO_ROOT, "best_deberta_large_curated_scorer")
        else:
            model_dir = DEFAULT_MODEL_DIR

    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found at {args.test_file}")
        sys.exit(1)

    print(f"Loading test dataset from: {args.test_file}...")
    df = pd.read_json(args.test_file, lines=True)
    texts = df["text"].tolist()
    y_true = np.array(df["base_score"].tolist(), dtype=float)
    strata = df["strata"].tolist() if "strata" in df.columns else None
    sources = df["source"].tolist() if "source" in df.columns else None

    print(f"Loading model from: {model_dir}...")
    scorer = PromptScorer(model_dir=model_dir, device=args.device)
    device_name = f"{scorer.device} ({torch.cuda.get_device_name(0) if scorer.device.type == 'cuda' else 'CPU'})"
    print(f"Initialized PromptScorer on {device_name}")

    print(f"Scoring {len(texts):,} test prompts in batches of {args.batch_size}...")
    start_time = time.time()
    results_list = scorer.score_batch(texts, batch_size=args.batch_size)
    elapsed = time.time() - start_time

    y_pred = np.array([r.score for r in results_list], dtype=float)

    timing_info = {
        "elapsed_sec": elapsed,
        "throughput": len(texts) / elapsed,
        "latency_ms": (elapsed / len(texts)) * 1000.0,
        "device": device_name,
        "batch_size": args.batch_size,
    }

    metrics = compute_all_metrics(y_true, y_pred, strata=strata, sources=sources)
    metrics["timing"] = timing_info
    metrics["model_dir"] = model_dir
    metrics["test_file"] = args.test_file

    print_evaluation_report(metrics, timing_info, model_name=os.path.basename(model_dir), test_file=args.test_file)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\u2714 Saved complete metrics JSON to: {args.output_json}")

    if args.save_predictions:
        preds_df = df.copy()
        preds_df["predicted_score"] = y_pred
        preds_df["predicted_bracket"] = [r.bracket for r in results_list]
        preds_df["residual_error"] = y_pred - y_true
        preds_df.to_json(args.save_predictions, orient="records", lines=True)
        print(f"\u2714 Saved per-sample predictions to: {args.save_predictions}")


if __name__ == "__main__":
    import torch
    main()
