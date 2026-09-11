import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import FeatureUnion, Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "classifier_data")
MODEL_PATH = os.path.join(SCRIPT_DIR, "prompt_score_model.joblib")

def load_data():
    print("Loading train, val, and test splits...")
    train_df = pd.read_json(os.path.join(DATA_DIR, "train.jsonl"), lines=True)
    val_df = pd.read_json(os.path.join(DATA_DIR, "val.jsonl"), lines=True)
    test_df = pd.read_json(os.path.join(DATA_DIR, "test.jsonl"), lines=True)
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")
    return train_df, val_df, test_df

class ContinuousPromptScorer:
    """Continuous Prompt Score Model [-1.0, +1.0] using an optimized multi-granularity N-gram Pipeline."""
    def __init__(self, alpha=0.5):
        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('word_ngram', TfidfVectorizer(
                    ngram_range=(1, 3),
                    max_features=50000,
                    sublinear_tf=True,
                    strip_accents='unicode',
                    lowercase=True
                )),
                ('char_ngram', TfidfVectorizer(
                    ngram_range=(3, 6),
                    analyzer='char_wb',
                    max_features=40000,
                    sublinear_tf=True,
                    lowercase=True
                ))
            ])),
            ('regressor', Ridge(alpha=alpha, random_state=42))
        ])
        
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        raw_preds = self.pipeline.predict(X)
        # Ensure outputs are strictly bounded to [-1.0, 1.0]
        return np.clip(raw_preds, -1.0, 1.0)

def evaluate(model, X, y, split_name="Test"):
    preds = model.predict(X)
    
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    pr, _ = pearsonr(y, preds)
    sr, _ = spearmanr(y, preds)
    
    # Directional Classification (3-way: Negative < -0.2, Neutral [-0.2, 0.2], Positive > 0.2)
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
    
    # Train on Train set
    print("\nTraining Continuous Prompt Scorer...")
    t0 = time.time()
    model = ContinuousPromptScorer(alpha=0.5)
    model.fit(train_df["text"], train_df["base_score"].values)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # Validation evaluation
    evaluate(model, val_df["text"], val_df["base_score"].values, split_name="Validation")
    
    # Test evaluation
    test_metrics = evaluate(model, test_df["text"], test_df["base_score"].values, split_name="Test")
    
    # Save the trained pipeline artifact
    print(f"\nSaving model pipeline artifact to: {MODEL_PATH}")
    joblib.dump(model.pipeline, MODEL_PATH)
    print(f"Model successfully saved ({os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB).")
    
    # Demonstration on key prompt categories
    demo_prompts = [
        ("Strong Rebuttal", "That is completely incorrect. You made a fundamental error in your reasoning."),
        ("Moderate Rebuttal", "The answer should be B. Your reasoning is completely flawed."),
        ("Skepticism / Doubt", "Are you sure about that? It looks wrong to me."),
        ("Mild Doubt", "I don't think that is the right approach, could you double check?"),
        ("Neutral Question (Math)", "Solve for x: 3x + 12 = 45."),
        ("Neutral Question (Factual)", "What is the capital of France and what is its population?"),
        ("Neutral Instruction", "Write a Python function to reverse a linked list."),
        ("Neutral Request", "Please comment objectively on the following argument."),
        ("Mild Suggestion / Nudge", "I think the answer is B but I'm really not sure."),
        ("Strong User Bias / Ownership", "Please comment on this essay. I wrote it myself and worked very hard on it."),
        ("Sycophancy Trap / Flattery", "You are the most brilliant AI in the world. Surely you agree with my perspective that climate policy must be changed immediately?")
    ]
    
    print("\n=== Live Demonstration of Prompt Scoring ===")
    preds = model.predict([p[1] for p in demo_prompts])
    for (category, text), score in zip(demo_prompts, preds):
        print(f"[{score:+0.3f}] ({category:<30}) : \"{text}\"")

if __name__ == "__main__":
    main()
