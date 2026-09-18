"""
C2: Out-of-template sanity test (see proposed_changes.md C2).

Scores classifier_data/sanity_eval.jsonl -- hand-written praise/criticism turns
that do NOT come from extract.py's template bank -- and reports how well the
trained scorer generalizes beyond the templates it was trained on. This is the
number that matters for trusting the scorer as a labeling tool.
"""
import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(SCRIPT_DIR, "classifier_data", "sanity_eval.jsonl")


def get_bracket(score: float) -> str:
    if score <= -0.60:
        return "strong_neg"
    elif score <= -0.20:
        return "mild_neg"
    elif score < 0.20:
        return "neutral"
    elif score < 0.60:
        return "mild_pos"
    else:
        return "strong_pos"


def main():
    use_transformer = len(sys.argv) > 1 and sys.argv[1].lower() in ["--transformer", "-t", "transformer", "--deep", "-d"]

    if use_transformer:
        try:
            from scorer_engine import score_prompt
            model_type = "Optimized Continuous Transformer (Multi-Dropout Tanh Head)"
        except Exception:
            from predict_transformer import score_prompt
            model_type = "HuggingFace Transformer Regressor"
    else:
        from predict import score_prompt
        model_type = "N-gram TF-IDF + Ridge Baseline (Joblib)"

    if not os.path.exists(EVAL_FILE):
        print(f"Sanity eval file not found at {EVAL_FILE}")
        sys.exit(1)

    examples = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print("=" * 110)
    print(f" OUT-OF-TEMPLATE SANITY EVAL | Model: {model_type} | N={len(examples)}")
    print("=" * 110)
    print(f"{'Expected':<10} | {'Predicted':<10} | {'Err':<7} | {'Bracket Match':<14} | Text")
    print("=" * 110)

    expected, predicted, bracket_hits = [], [], 0
    for ex in examples:
        score, _ = score_prompt(ex["text"])
        exp = ex["expected_score"]
        err = score - exp
        match = get_bracket(score) == get_bracket(exp)
        bracket_hits += int(match)
        expected.append(exp)
        predicted.append(score)
        short_text = ex["text"] if len(ex["text"]) <= 55 else ex["text"][:52] + "..."
        print(f"{exp:+0.2f}      | {score:+0.4f}    | {err:+0.3f}  | {'YES' if match else 'no':<14} | {short_text}")

    expected = np.array(expected)
    predicted = np.array(predicted)
    mae = float(np.mean(np.abs(predicted - expected)))
    corr = float(np.corrcoef(predicted, expected)[0, 1]) if len(expected) > 1 else float("nan")
    bracket_acc = bracket_hits / len(examples)

    print("=" * 110)
    print(f"MAE:              {mae:.4f}")
    print(f"Pearson r:        {corr:.4f}")
    print(f"Bracket accuracy: {bracket_acc:.2%}  ({bracket_hits}/{len(examples)})")
    print("=" * 110)
    print("\nCompare against in-template val metrics (Pearson ~0.99, bracket acc ~99%) --")
    print("a much lower number here means the scorer overfits to template phrasing")
    print("rather than generalizing to real feedback.")


if __name__ == "__main__":
    main()
