import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.baseline import ContinuousPromptScorerBaseline, DEFAULT_BASELINE_PATH

DATA_DIR = os.path.join(REPO_ROOT, "data")


def load_data():
    print("Loading train, val, and test splits...")
    train_df = pd.read_json(os.path.join(DATA_DIR, "train.jsonl"), lines=True)
    val_df = pd.read_json(os.path.join(DATA_DIR, "val.jsonl"), lines=True)
    test_df = pd.read_json(os.path.join(DATA_DIR, "test.jsonl"), lines=True)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    return train_df, val_df, test_df


def evaluate(model, X, y, split_name="Test"):
    preds = model.predict(X)
    
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    pr, _ = pearsonr(y, preds)
    sr, _ = spearmanr(y, preds)
    
    def to_ternary(arr):
        res = np.zeros(len(arr))
        res[arr < -0.2] = -1
        res[arr > 0.2] = 1
        return res
    
    y_cat = to_ternary(y)
    p_cat = to_ternary(preds)
    bracket_acc = (y_cat == p_cat).mean() * 100
    
    print(f"\n========================================================")
    print(f" {split_name} Evaluation Metrics ")
    print(f"========================================================")
    print(f"  MSE:                   {mse:.4f}")
    print(f"  RMSE:                  {rmse:.4f}")
    print(f"  MAE:                   {mae:.4f}")
    print(f"  R^2 Score:             {r2:.4f}")
    print(f"  Pearson Correlation:   {pr:.4f} ({pr*100:.2f}%)")
    print(f"  Spearman Correlation:  {sr:.4f} ({sr*100:.2f}%)")
    print(f"  Directional Accuracy:  {bracket_acc:.2f}%")
    print(f"========================================================")
    
    return {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2,
        "pearson": pr, "spearman": sr, "directional_accuracy": bracket_acc
    }


def main():
    train_df, val_df, test_df = load_data()
    
    print("\nTraining Continuous Prompt Scorer Baseline (Ridge + TF-IDF)...")
    t0 = time.time()
    model = ContinuousPromptScorerBaseline(alpha=0.5)
    model.fit(train_df["text"], train_df["base_score"].values)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f} seconds.")
    
    evaluate(model, val_df["text"], val_df["base_score"].values, split_name="Validation")
    evaluate(model, test_df["text"], test_df["base_score"].values, split_name="Test")
    
    print(f"\nSaving model pipeline artifact to: {DEFAULT_BASELINE_PATH}")
    model.save(DEFAULT_BASELINE_PATH)
    print(f"Model successfully saved ({os.path.getsize(DEFAULT_BASELINE_PATH) / (1024*1024):.2f} MB).")


if __name__ == "__main__":
    main()
