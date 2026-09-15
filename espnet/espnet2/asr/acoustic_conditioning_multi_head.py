import torch
import torch.nn as nn


class MultiHeadAcousticConditioning(nn.Module):
    """Multi-Head Frequency-Band Acoustic Conditioning.

    input_dim (e.g. 80) fbank feature를
    num_heads개 주파수 대역으로 분할하여
    각 대역에 독립적인 conditioning을 적용한다.

    Interface:
        AcousticConditioning과 동일한 forward signature를 유지하여
        espnet_model.py에서 drop-in replacement로 사용 가능.
    """

    def __init__(
        self,
        acoustic_dim: int,
        input_dim: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()

        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim ({input_dim}) must be divisible "
                f"by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Per-head condition projection
        # acoustic_dim -> hidden_dim -> head_dim
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(acoustic_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, self.head_dim),
                )
                for _ in range(num_heads)
            ]
        )

        # Per-head channel-wise gate
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(acoustic_dim, self.head_dim),
                    nn.Sigmoid(),
                )
                for _ in range(num_heads)
            ]
        )

        # Per-head learnable residual scale
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_heads)]
        )

    def forward(
        self,
        feats,
        acoustic_feat,
        return_condition_vector: bool = False,
    ):
        """
        Args:
            feats:         [B, T, input_dim]
            acoustic_feat: [B, acoustic_dim]

        Returns:
            conditioned_feats: [B, T, input_dim]
            condition_vector:  [B, input_dim]  (if return_condition_vector)
        """

        # [B, T, input_dim] -> num_heads x [B, T, head_dim]
        feat_chunks = feats.chunk(self.num_heads, dim=-1)

        conditioned_chunks = []
        cond_vectors = []

        for i in range(self.num_heads):
            cond = self.nets[i](acoustic_feat)   # [B, head_dim]
            gate = self.gates[i](acoustic_feat)  # [B, head_dim]

            cond_expand = cond.unsqueeze(1)  # [B, 1, head_dim]
            gate_expand = gate.unsqueeze(1)  # [B, 1, head_dim]

            conditioned = (
                feat_chunks[i]
                + self.alphas[i] * gate_expand * cond_expand
            )

            conditioned_chunks.append(conditioned)
            cond_vectors.append(cond)

        # num_heads x [B, T, head_dim] -> [B, T, input_dim]
        conditioned_feats = torch.cat(conditioned_chunks, dim=-1)

        if return_condition_vector:
            # num_heads x [B, head_dim] -> [B, input_dim]
            condition_vector = torch.cat(cond_vectors, dim=-1)
            return conditioned_feats, condition_vector

        return conditioned_feats
