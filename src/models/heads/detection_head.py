"""Detection head placeholder."""

import torch
from torch import nn


class DetectionHead(nn.Module):
    """Placeholder detection head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Identity()

    def forward(self, features):
        """Forward pass."""
        return self.head(features)
