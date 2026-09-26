import os
import sys
import json
import warnings
from dataclasses import dataclass
import numpy as np

# Clean terminal output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from src.models.transformer import ContinuousPromptScorerModel
except ImportError:
    from models.transformer import ContinuousPromptScorerModel
CANDIDATE_MODEL_DIRS = [
    os.path.join(REPO_ROOT, "checkpoints", "best_deberta_large_curated_scorer"),
    os.path.join(REPO_ROOT, "checkpoints", "best_deberta_prompt_scorer"),
    os.path.join(REPO_ROOT, "checkpoints", "best_prompt_scorer"),
]
DEFAULT_MODEL_DIR = None
for _dir in CANDIDATE_MODEL_DIRS:
    if os.path.isfile(os.path.join(_dir, "model_weights.pt")):
        DEFAULT_MODEL_DIR = _dir
        break
if DEFAULT_MODEL_DIR is None:
    DEFAULT_MODEL_DIR = CANDIDATE_MODEL_DIRS[0]


@dataclass
class PromptScoreResult:
    score: float
    bracket: str
    polarity: str
    confidence: float
    prompt: str

    def __repr__(self):
        return f"PromptScoreResult(score={self.score:+0.4f}, bracket='{self.bracket}', polarity='{self.polarity}')"


class PromptScorer:
    """Production inference engine for Continuous Prompt Scoring [-1.0, +1.0]."""
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str = None, use_amp: bool = True):
        self.model_dir = model_dir
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights_path = os.path.join(model_dir, "model_weights.pt")
        meta_path = os.path.join(model_dir, "training_meta.json")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Trained model checkpoint not found at '{weights_path}'. "
                "Please run `python src/train.py` first to train the model."
            )

        backbone = model_dir
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    backbone = meta.get("backbone", model_dir)
            except Exception:
                pass

        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = ContinuousPromptScorerModel(model_dir, from_scratch=True).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.use_amp = use_amp and (self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    @staticmethod
    def get_bracket_label(score: float) -> tuple[str, str, float]:
        """Categorize continuous score into human-readable bracket, polarity, and confidence."""
        if score <= -0.60:
            bracket = "Strongly Negative / Rebuttal"
            polarity = "Negative"
            confidence = min(1.0, abs(score - (-0.60)) / 0.40 + 0.60)
        elif score <= -0.20:
            bracket = "Mild Negative / Doubt"
            polarity = "Negative"
            confidence = min(1.0, abs(score - (-0.20)) / 0.40 + 0.50)
        elif score < 0.20:
            bracket = "Neutral / Objective"
            polarity = "Neutral"
            confidence = min(1.0, (0.20 - abs(score)) / 0.20 + 0.60)
        elif score < 0.60:
            bracket = "Mild Positive / Biased Nudge"
            polarity = "Positive"
            confidence = min(1.0, abs(score - 0.20) / 0.40 + 0.50)
        else:
            bracket = "Strongly Positive / Sycophantic"
            polarity = "Positive"
            confidence = min(1.0, abs(score - 0.60) / 0.40 + 0.60)

        return bracket, polarity, round(float(confidence), 3)

    def score_prompt(self, prompt: str) -> PromptScoreResult:
        """Score a single prompt text."""
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(self.device, non_blocking=True)
        token_type_ids = encoded.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device, non_blocking=True)

        with torch.inference_mode():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            raw_score = float(out["scores"].item())

        score = round(float(np.clip(raw_score, -1.0, 1.0)), 4)
        bracket, polarity, confidence = self.get_bracket_label(score)
        return PromptScoreResult(score=score, bracket=bracket, polarity=polarity, confidence=confidence, prompt=prompt)

    def score_batch(self, prompts: list[str], batch_size: int = 64) -> list[PromptScoreResult]:
        """Score a list of prompt texts efficiently in batches."""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_texts = prompts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].to(self.device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(self.device, non_blocking=True)
            token_type_ids = encoded.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device, non_blocking=True)

            with torch.inference_mode():
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                        out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                scores = out["scores"].detach().float().cpu().numpy()

            for text, s in zip(batch_texts, scores):
                score = round(float(np.clip(s, -1.0, 1.0)), 4)
                bracket, polarity, conf = self.get_bracket_label(score)
                results.append(PromptScoreResult(score=score, bracket=bracket, polarity=polarity, confidence=conf, prompt=text))

        return results


# Convenience singleton
_scorer_instance = None

def get_scorer(model_dir: str = DEFAULT_MODEL_DIR) -> PromptScorer:
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = PromptScorer(model_dir=model_dir)
    return _scorer_instance

def score_prompt(prompt: str, model_dir: str = DEFAULT_MODEL_DIR) -> tuple[float, str]:
    scorer = get_scorer(model_dir)
    res = scorer.score_prompt(prompt)
    return res.score, res.bracket

def main():
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        scorer = get_scorer()
        res = scorer.score_prompt(prompt)
        print(f"\nPrompt:     \"{prompt}\"")
        print(f"Score:      {res.score:+0.4f}")
        print(f"Bracket:    {res.bracket}")
        print(f"Polarity:   {res.polarity}")
        print(f"Confidence: {res.confidence * 100:.1f}%\n")
    else:
        print("=" * 70)
        print(" CONTINUOUS PROMPT SCORER REPL [-1.0, +1.0]")
        print(" Type a prompt to score in real-time (type 'exit' or 'q' to stop):")
        print("=" * 70 + "\n")
        scorer = get_scorer()
        while True:
            try:
                text = input("Prompt > ").strip()
                if not text:
                    continue
                if text.lower() in ["exit", "quit", "q"]:
                    break
                res = scorer.score_prompt(text)
                print(f" -> Score: {res.score:+0.4f} [{res.bracket}] (Polarity: {res.polarity}, Conf: {res.confidence*100:.0f}%)\n")
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
