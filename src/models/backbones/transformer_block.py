"""Minimal Transformer block for mid/high-level feature refinement."""

from __future__ import annotations

import torch
from torch import nn


class SimpleTransformerBlock(nn.Module):
    """A small self-attention block for a single feature map level."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Transformer dim={dim} must be divisible by num_heads={num_heads}.")

        hidden_dim = max(int(dim * mlp_ratio), dim)
        self.positional_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine a 2D feature map with lightweight global attention."""
        batch_size, channels, height, width = x.shape
        x = x + self.positional_conv(x)
        tokens = x.flatten(2).transpose(1, 2)

        attn_input = self.norm1(tokens)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_output
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)
