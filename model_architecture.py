import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultiSampleDropoutHead(nn.Module):
    """
    Multi-Sample Dropout 2-Stage MLP Regression Head with Tanh bounding.
    Averages predictions across multiple dropout masks during training
    to accelerate convergence and improve generalization.
    """
    def __init__(self, hidden_size: int, intermediate_size: int = 256, dropout_rate: float = 0.2, num_samples: int = 5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(intermediate_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features shape: (batch_size, hidden_size)
        normed = self.layer_norm(features)
        
        if self.training:
            # Multi-sample dropout passes
            sample_outputs = []
            for drop in self.dropouts:
                h = self.activation(self.dense(drop(normed)))
                out = self.out_proj(h)
                sample_outputs.append(out)
            # Average the logits before tanh
            mean_logits = torch.mean(torch.stack(sample_outputs, dim=0), dim=0)
        else:
            h = self.activation(self.dense(normed))
            mean_logits = self.out_proj(h)
            
        # Strictly bound output to [-1.0, +1.0] via Tanh
        return self.tanh(mean_logits).squeeze(-1)


class HybridPromptScoreLoss(nn.Module):
    """
    Multi-objective Loss combining Smooth L1 (Huber) regression loss
    with Pearson Correlation ranking loss.
    L_total = SmoothL1(y_pred, y_true) + alpha * (1 - Pearson(y_pred, y_true))
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # 1. Smooth L1 (Metric accuracy)
        l1_loss = self.smooth_l1(y_pred, y_true)
        
        # 2. Pearson Correlation Loss (Ordinal Ranking)
        if len(y_pred) > 2:
            pred_centered = y_pred - torch.mean(y_pred)
            true_centered = y_true - torch.mean(y_true)
            
            covariance = torch.sum(pred_centered * true_centered)
            pred_std = torch.sqrt(torch.sum(pred_centered ** 2) + 1e-8)
            true_std = torch.sqrt(torch.sum(true_centered ** 2) + 1e-8)
            
            pearson_r = covariance / (pred_std * true_std + 1e-8)
            # Pearson loss is in [0, 2] where 0 is perfect correlation
            pearson_loss = 1.0 - pearson_r
        else:
            pearson_loss = torch.tensor(0.0, device=y_pred.device)

        return l1_loss + self.alpha * pearson_loss


class ContinuousPromptScorerModel(nn.Module):
    """
    Continuous Prompt Scorer [-1.0, +1.0] Model.
    Combines a Transformer backbone (e.g. DeBERTa-v3 or MiniLM)
    with Attention Pooling and a Multi-Sample Dropout Tanh Head.
    """
    def __init__(self, model_name_or_path: str, dropout_rate: float = 0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config).float()
        
        hidden_size = self.config.hidden_size
        
        # Attention Pooling Layer
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Bounded Regression Head
        self.head = MultiSampleDropoutHead(
            hidden_size=hidden_size,
            intermediate_size=256,
            dropout_rate=dropout_rate,
            num_samples=5
        )

    def pool_tokens(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling across sequence tokens, respecting padding mask."""
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)
        
        # Calculate raw attention scores
        attn_scores = self.attention_weights(last_hidden_state).squeeze(-1) # (batch, seq_len)
        
        # Mask padded positions with FP16-safe large negative value (-10000.0)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, -10000.0)
        attn_probs = torch.softmax(attn_scores, dim=-1).unsqueeze(-1) # (batch, seq_len, 1)
        
        # Weighted sum of token representations
        pooled = torch.sum(last_hidden_state * attn_probs, dim=1) # (batch, hidden_size)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.config.to_dict():
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)
        last_hidden_state = outputs.last_hidden_state
        
        # Pool contextual representations
        pooled = self.pool_tokens(last_hidden_state, attention_mask)
        
        # Compute strictly bounded prompt score [-1.0, 1.0]
        scores = self.head(pooled)
        
        result = {"scores": scores}
        if labels is not None:
            loss_fn = HybridPromptScoreLoss(alpha=0.5)
            loss = loss_fn(scores, labels.float())
            result["loss"] = loss
            
        return result
