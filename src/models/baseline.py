import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASELINE_PATH = os.path.join(REPO_ROOT, "checkpoints", "prompt_score_model.joblib")


class ContinuousPromptScorerBaseline:
    """Continuous Prompt Score Baseline [-1.0, +1.0] using multi-granularity N-gram Pipeline."""
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
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
            ('regressor', Ridge(alpha=self.alpha, random_state=42))
        ])
        
    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        raw_preds = self.pipeline.predict(X)
        return np.clip(raw_preds, -1.0, 1.0)

    def save(self, filepath: str = DEFAULT_BASELINE_PATH):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)

    @classmethod
    def load(cls, filepath: str = DEFAULT_BASELINE_PATH):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Baseline model not found at '{filepath}'.")
        inst = cls()
        inst.pipeline = joblib.load(filepath)
        return inst
