"""Detection loss placeholder."""

from typing import Dict
import torch


def compute_detection_loss(predictions, targets) -> Dict[str, torch.Tensor]:
    """Return placeholder detection losses."""
    # TODO: implement actual detection losses
    loss = torch.tensor(0.0, requires_grad=True)
    return {"loss_total": loss}
