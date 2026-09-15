import torch
import torch.nn as nn


class AcousticConditioning(nn.Module):
    def __init__(
        self,
        acoustic_dim: int,
        input_dim: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(acoustic_dim, input_dim),
            nn.Sigmoid(),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, feats, acoustic_feat):

        cond = self.net(acoustic_feat)
        gate = self.gate(acoustic_feat)

        cond = cond.unsqueeze(1)
        gate = gate.unsqueeze(1)

        return feats + self.alpha * gate * cond
