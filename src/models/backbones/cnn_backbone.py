"""Lightweight CNN backbone placeholder."""

import torch
from torch import nn


class CNNBackbone(nn.Module):
    """A simple CNN-dominant backbone placeholder."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Identity()

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return {"p2": x, "p3": x, "p4": x, "p5": x}
