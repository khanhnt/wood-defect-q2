"""Loss helpers for the minimal baseline detector pipeline."""

from __future__ import annotations

from typing import Dict

import torch


def compute_detection_loss(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reduce a torchvision-style loss dictionary into a stable scalar summary."""
    if not isinstance(loss_dict, dict):
        raise TypeError("Expected a loss dictionary from the detector model.")

    reduced = {name: value for name, value in loss_dict.items()}
    loss_total = sum(reduced.values()) if reduced else torch.tensor(0.0)
    reduced["loss_total"] = loss_total
    return reduced


def detach_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Detach a loss dictionary for logging."""
    return {
        name: round(float(value.detach().cpu()), 6)
        for name, value in loss_dict.items()
    }
