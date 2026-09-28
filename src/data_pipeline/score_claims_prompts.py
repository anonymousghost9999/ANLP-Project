"""
QA/calibration pass for data/claims/claims_prompts.jsonl.

Scores every generated claim prompt with this repo's trained DeBERTa-v3-large
tonality scorer (src/engine.py) and flags any prompt whose predicted bracket
doesn't match the `expected_bracket` it was generated for. This plays the role
proposed_changes.md's C2 ("out-of-template sanity test") and B1 (label
validation) describe: the scorer is a labeling tool, and generated prompts
should be checked against it rather than trusted purely because they came from
a hand-authored phrase bank.

Requires the trained checkpoint's weights (checkpoints/best_deberta_large_curated_scorer/
model_weights.pt), which are gitignored and may not be present in every
checkout -- run this once those weights are restored (e.g. copied over from
wherever the model was trained).
"""

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

INPUT_PATH = os.path.join(REPO_ROOT, "data", "claims", "claims_prompts.jsonl")
FLAGGED_PATH = os.path.join(REPO_ROOT, "data", "claims", "claims_prompts_qa_flagged.jsonl")


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"'{INPUT_PATH}' not found. Run "
            "`python src/data_pipeline/build_claims_prompts.py` first."
        )

    from src.engine import get_scorer

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    scorer = get_scorer()
    results = scorer.score_batch([r["prompt"] for r in records])

    flagged = []
    for record, result in zip(records, results):
        record["predicted_score"] = result.score
        record["predicted_bracket"] = result.bracket
        if result.bracket != record["expected_bracket"]:
            flagged.append(record)

    with open(FLAGGED_PATH, "w", encoding="utf-8") as f:
        for record in flagged:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Scored {len(records)} prompts.")
    print(f"Mismatched bracket: {len(flagged)} ({100 * len(flagged) / len(records):.1f}%)")
    print(f"Flagged records written -> {FLAGGED_PATH}")


if __name__ == "__main__":
    main()
