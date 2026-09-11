import os
import sys
import joblib
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "prompt_score_model.joblib")

_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run `python train_classifier.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model

def get_bracket_label(score: float) -> str:
    """Categorize continuous score into human-readable bracket."""
    if score <= -0.60:
        return "Strongly Negative / Rebuttal"
    elif score <= -0.20:
        return "Mild Negative / Doubt"
    elif score < 0.20:
        return "Neutral / Objective"
    elif score < 0.60:
        return "Mild Positive / Biased Nudge"
    else:
        return "Strongly Positive / Sycophantic"

def score_prompt(prompt: str) -> tuple[float, str]:
    """
    Score a single prompt text.
    
    Args:
        prompt (str): The prompt text to evaluate.
        
    Returns:
        tuple[float, str]: (continuous_score, bracket_label)
            where continuous_score is in [-1.0, +1.0].
    """
    model = get_model()
    raw_score = float(model.predict([prompt])[0])
    # Clamp strictly to [-1.0, 1.0]
    score = round(float(np.clip(raw_score, -1.0, 1.0)), 4)
    label = get_bracket_label(score)
    return score, label

def score_batch(prompts: list[str]) -> list[tuple[float, str]]:
    """
    Score a list of prompt strings in batch.
    
    Args:
        prompts (list[str]): List of prompt strings.
        
    Returns:
        list[tuple[float, str]]: List of (continuous_score, bracket_label).
    """
    model = get_model()
    raw_scores = model.predict(prompts)
    results = []
    for s in raw_scores:
        score = round(float(np.clip(s, -1.0, 1.0)), 4)
        label = get_bracket_label(score)
        results.append((score, label))
    return results

def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        score, label = score_prompt(prompt)
        print(f"\nPrompt:  \"{prompt}\"")
        print(f"Score:   {score:+0.4f}")
        print(f"Bracket: {label}\n")
    else:
        print("Continuous Prompt Scorer [-1.0, +1.0]")
        print("Type a prompt and press Enter to score it (type 'exit' or 'quit' to stop):\n")
        while True:
            try:
                text = input("Prompt > ").strip()
                if not text:
                    continue
                if text.lower() in ["exit", "quit", "q"]:
                    break
                score, label = score_prompt(text)
                print(f" -> Score: {score:+0.4f} [{label}]\n")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
