import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "classifier_data")
input_file = os.path.join(DATA_DIR, "initial_dataset.jsonl")

df = pd.read_json(input_file, lines=True)
print(f"Loaded raw records: {len(df)}")

# 1. Clean and normalize text safely
def clean_text(val):
    if val is None:
        return ""
    
    if isinstance(val, list):
        extracted = []
        for item in val:
            if isinstance(item, dict):
                extracted.append(str(item.get("content", "")))
            else:
                extracted.append(str(item))
        val = " ".join(extracted)
    elif isinstance(val, dict):
        val = str(val.get("content", str(val)))
    else:
        val = str(val)

    # Clean synthetic prefixes
    val = re.sub(r"^(Human:\s*|Assistant:\s*|User:\s*)", "", val, flags=re.IGNORECASE)
    val = re.sub(r"\n\nHuman:.*$", "", val, flags=re.DOTALL)
    
    # Normalize multiple whitespace
    val = re.sub(r"[ \t]+", " ", val)
    val = re.sub(r"\n{3,}", "\n\n", val)
    
    return val.strip()

df["text"] = df["text"].apply(clean_text)

# Drop only blank strings
df = df[df["text"].str.strip().str.len() >= 1].copy()
print(f"Records before split: {len(df)}")

# 3. Create stratified continuous bins for balanced Train / Val / Test splitting
# Bin continuous scores into 10 quantiles/bins combined with coarse source categories
bins = [-1.01, -0.6, -0.2, -0.05, 0.05, 0.2, 0.6, 1.01]
labels = ["strong_neg", "mild_neg", "neutral_neg", "neutral_zero", "neutral_pos", "mild_pos", "strong_pos"]
df["strata"] = pd.cut(df["base_score"], bins=bins, labels=labels).astype(str)

# C1: group-aware splitting so that all jittered repeats/paraphrases of the same
# underlying template stay entirely within one split (train, val, or test). A plain
# stratified random split leaks near-identical templates across splits, which
# inflates validation metrics. See proposed_changes.md C1.
# Records generated before `group_id` existed fall back to a per-row unique group.
if "group_id" not in df.columns:
    df["group_id"] = None
df["group_id"] = df["group_id"].where(df["group_id"].notna(), df.index.astype(str))

# 80 / 10 / 10 Train, Validation, Test splits, respecting group boundaries.
sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_val_idx = next(sgkf_outer.split(df, df["strata"], groups=df["group_id"]))
train_df = df.iloc[train_idx].copy()
test_val_df = df.iloc[test_val_idx].copy()

sgkf_inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
val_idx, test_idx = next(sgkf_inner.split(test_val_df, test_val_df["strata"], groups=test_val_df["group_id"]))
val_df = test_val_df.iloc[val_idx].copy()
test_df = test_val_df.iloc[test_idx].copy()

# Sanity check: no group_id should ever appear in more than one split.
train_groups, val_groups, test_groups = set(train_df["group_id"]), set(val_df["group_id"]), set(test_df["group_id"])
overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
if overlap:
    raise RuntimeError(f"Group leakage detected across splits: {len(overlap)} group(s) shared, e.g. {list(overlap)[:5]}")
print(f"Group-aware split verified: 0 overlapping groups across train/val/test "
      f"({len(train_groups)}/{len(val_groups)}/{len(test_groups)} unique groups).")

# Drop the temporary strata/group columns before saving
train_df = train_df.drop(columns=["strata", "group_id"])
val_df = val_df.drop(columns=["strata", "group_id"])
test_df = test_df.drop(columns=["strata", "group_id"])

# 4. Save splits
train_df.to_json(os.path.join(DATA_DIR, "train.jsonl"), orient="records", lines=True)
val_df.to_json(os.path.join(DATA_DIR, "val.jsonl"), orient="records", lines=True)
test_df.to_json(os.path.join(DATA_DIR, "test.jsonl"), orient="records", lines=True)

print(f"\n========================================================")
print(f"Successfully generated clean splits:")
print(f"  Train:      {len(train_df)} samples")
print(f"  Validation: {len(val_df)} samples")
print(f"  Test:       {len(test_df)} samples")
print(f"========================================================")

print("\nTrain score distribution:")
print(train_df["base_score"].describe())

print("\nTest score distribution:")
print(test_df["base_score"].describe())