"""Lightweight Transformer block placeholder."""

import torch
from torch import nn


class SimpleTransformerBlock(nn.Module):
    """A simple placeholder Transformer block."""

    def __init__(self, dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.block = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)
