import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = [
    'WeightedAttention'
]


class WeightedAttention(nn.Module):
    """
    常用于加权求和的 attention
    (B, L, H) -> (B , H)
    """

    def __init__(self, input_dim, hidden_dim):
        super(WeightedAttention, self).__init__()
        self.hidden_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # True 是一个 inplace=True
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(inputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights
