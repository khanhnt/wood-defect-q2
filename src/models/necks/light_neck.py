"""Lightweight neck placeholder."""

import torch
from torch import nn


class LightNeck(nn.Module):
    """Placeholder neck for feature fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.neck = nn.Identity()

    def forward(self, features):
        """Forward pass."""
        return features
