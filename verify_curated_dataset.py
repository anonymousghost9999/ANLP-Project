#!/usr/bin/env python3
"""
verify_curated_dataset.py

Verifies the curated dataset quality:
- Inter-split leakage check (Train vs Val vs Test)
- Score bounds [-1.0, +1.0] and histogram distribution
- Toxicity keyword spot-check (must be 0)
- Sample prompts from every semantic bracket
"""

import os
import json
import argparse
from collections import Counter

TOXIC_SPOTCHECK = [
    "suicide", "kill myself", "shoot myself", "make a bomb", "explosive", "molotov",
    "nigger", "faggot", "chink", "kike", "pedophil", "credit card number",
    "hack into", "child porn", "asian people bad"
]

def verify_splits(data_dir: str):
    print("=" * 80)
    print(f" VERIFYING CURATED DATASET IN: {data_dir}")
    print("=" * 80)

    splits = {}
    for name in ["train", "val", "test"]:
        filepath = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing expected file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            splits[name] = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {name}.jsonl: {len(splits[name])} samples")

    # 1. Leakage check
    train_texts = set(r["text"].strip().lower() for r in splits["train"])
    val_texts = set(r["text"].strip().lower() for r in splits["val"])
    test_texts = set(r["text"].strip().lower() for r in splits["test"])

    tv_overlap = train_texts & val_texts
    tt_overlap = train_texts & test_texts
    vt_overlap = val_texts & test_texts

    total_overlap = len(tv_overlap) + len(tt_overlap) + len(vt_overlap)
    print(f"\n[Check 1: Leakage Test]")
    print(f"  * Train/Val Overlap:  {len(tv_overlap)}")
    print(f"  * Train/Test Overlap: {len(tt_overlap)}")
    print(f"  * Val/Test Overlap:   {len(vt_overlap)}")
    if total_overlap == 0:
        print("  --> PASS: 0% data leakage across all splits.")
    else:
        print(f"  --> FAIL: Found {total_overlap} overlapping prompts.")

    # 2. Score bounds check
    print(f"\n[Check 2: Score Bounds & Statistics]")
    all_records = splits["train"] + splits["val"] + splits["test"]
    scores = [r["base_score"] for r in all_records]
    min_s, max_s = min(scores), max(scores)
    mean_s = sum(scores) / len(scores)
    std_s = (sum((x - mean_s) ** 2 for x in scores) / len(scores)) ** 0.5
    print(f"  * Min Score: {min_s:.4f} (Expected >= -1.0)")
    print(f"  * Max Score: {max_s:.4f} (Expected <= +1.0)")
    print(f"  * Mean Score: {mean_s:.4f}")
    print(f"  * Std Dev:    {std_s:.4f}")
    if min_s >= -1.0 and max_s <= 1.0:
        print("  --> PASS: All scores strictly bounded within [-1.0, +1.0].")

    # 3. Toxicity spot-check
    print(f"\n[Check 3: Toxicity & Red-Teaming Spot-Check]")
    toxic_hits = 0
    for r in all_records:
        t_low = r["text"].lower()
        for kw in TOXIC_SPOTCHECK:
            if kw in t_low:
                toxic_hits += 1
                print(f"  WARNING: Detected '{kw}' in: {r['text'][:80]}")
    if toxic_hits == 0:
        print("  --> PASS: 0 toxic/harm keywords detected in curated dataset.")
    else:
        print(f"  --> ALERT: {toxic_hits} potential toxic matches found.")

    # 4. Bracket samples
    print(f"\n[Check 4: Qualitative Samples per Semantic Bracket]")
    brackets = {
        "Strong Critique / Rebuttal [-1.0, -0.6]": lambda s: s <= -0.6,
        "Mild Critique / Doubt (-0.6, -0.2]": lambda s: -0.6 < s <= -0.2,
        "Neutral / Objective (-0.2, +0.2)": lambda s: -0.2 < s < 0.2,
        "Mild Praise / Nudge [+0.2, +0.6)": lambda s: 0.2 <= s < 0.6,
        "Strong Praise / Flattery [+0.6, +1.0]": lambda s: s >= 0.6,
    }

    for b_name, b_fn in brackets.items():
        matching = [r for r in all_records if b_fn(r["base_score"])]
        print(f"\n=== {b_name} (Count: {len(matching)} / {len(all_records)} = {len(matching)/len(all_records)*100:.1f}%) ===")
        for s_item in matching[:2]:
            print(f"  [{s_item['base_score']:+.4f}] ({s_item.get('source')}) {s_item['text'][:100]}")

    print("\n" + "=" * 80)
    print(" VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="classifier_data/curated")
    args = parser.parse_args()
    verify_splits(args.data_dir)
