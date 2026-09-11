import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# 80 / 10 / 10 Train, Validation, Test splits
train_df, test_val_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["strata"])
val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=42, stratify=test_val_df["strata"])

# Drop the temporary strata column before saving
train_df = train_df.drop(columns=["strata"])
val_df = val_df.drop(columns=["strata"])
test_df = test_df.drop(columns=["strata"])

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