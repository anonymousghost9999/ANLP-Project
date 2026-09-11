import os
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuned_prompt_scorer")

_tokenizer = None
_model = None

def load_transformer(model_dir=DEFAULT_MODEL_DIR):
    global _tokenizer, _model
    if _model is None:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Model directory '{model_dir}' not found. "
                "Please run `python train_transformer_hpc.py` first."
            )
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        _model.eval()
        if torch.cuda.is_available():
            _model.to("cuda")
    return _tokenizer, _model

def get_bracket_label(score: float) -> str:
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

def score_prompt(prompt: str, model_dir=DEFAULT_MODEL_DIR) -> tuple[float, str]:
    tok, model = load_transformer(model_dir)
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        output = model(**inputs)
        raw_score = output.logits.item()
    score = round(float(np.clip(raw_score, -1.0, 1.0)), 4)
    label = get_bracket_label(score)
    return score, label

def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        score, label = score_prompt(prompt)
        print(f"\nPrompt:  \"{prompt}\"")
        print(f"Score:   {score:+0.4f}")
        print(f"Bracket: {label}\n")
    else:
        print("Transformer Continuous Prompt Scorer [-1.0, +1.0]")
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
