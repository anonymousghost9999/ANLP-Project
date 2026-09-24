"""
Continuous Prompt Scorer [-1.0, +1.0] Package.
Quantifies input prompts on a continuous scale from criticism/rebuttal (-1.0)
to neutral objective (0.0) to sycophancy/flattery (+1.0).
"""

from .models.transformer import ContinuousPromptScorerModel, HybridPromptScoreLoss, MultiSampleDropoutHead
from .models.baseline import ContinuousPromptScorerBaseline
from .engine import PromptScorer, PromptScoreResult, get_scorer, score_prompt

__all__ = [
    "ContinuousPromptScorerModel",
    "HybridPromptScoreLoss",
    "MultiSampleDropoutHead",
    "ContinuousPromptScorerBaseline",
    "PromptScorer",
    "PromptScoreResult",
    "get_scorer",
    "score_prompt",
]
