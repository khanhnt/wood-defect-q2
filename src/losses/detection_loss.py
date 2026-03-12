"""Loss helpers for the minimal detector pipeline."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute sigmoid focal loss for dense classification."""
    probabilities = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_weight = alpha_factor * torch.pow(1.0 - p_t, gamma)
    loss = ce_loss * focal_weight

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


def generalized_box_iou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute generalized IoU loss on XYXY boxes."""
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=-1)
    target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(dim=-1)

    inter_x1 = torch.maximum(pred_x1, target_x1)
    inter_y1 = torch.maximum(pred_y1, target_y1)
    inter_x2 = torch.minimum(pred_x2, target_x2)
    inter_y2 = torch.minimum(pred_y2, target_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    intersection = inter_w * inter_h

    pred_area = (pred_x2 - pred_x1).clamp(min=0.0) * (pred_y2 - pred_y1).clamp(min=0.0)
    target_area = (target_x2 - target_x1).clamp(min=0.0) * (target_y2 - target_y1).clamp(min=0.0)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp(min=eps)

    closure_x1 = torch.minimum(pred_x1, target_x1)
    closure_y1 = torch.minimum(pred_y1, target_y1)
    closure_x2 = torch.maximum(pred_x2, target_x2)
    closure_y2 = torch.maximum(pred_y2, target_y2)
    closure_area = (closure_x2 - closure_x1).clamp(min=0.0) * (closure_y2 - closure_y1).clamp(min=0.0)

    giou = iou - (closure_area - union) / closure_area.clamp(min=eps)
    loss = 1.0 - giou

    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


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
