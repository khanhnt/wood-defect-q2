"""WNIoU-style loss patching utilities for Ultralytics YOLO training."""

from __future__ import annotations

from typing import Any

import torch


def _xyxy_to_cxcywh(boxes_xyxy: torch.Tensor, eps: float = 1e-9) -> tuple[torch.Tensor, ...]:
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    widths = (x2 - x1).clamp_min(eps)
    heights = (y2 - y1).clamp_min(eps)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    return center_x, center_y, widths, heights


def normalized_wasserstein_similarity(
    pred_boxes_xyxy: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    *,
    distance_scale: float = 0.05,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute an NWD-style similarity for normalized axis-aligned boxes."""
    scale = max(float(distance_scale), eps)
    pred_cx, pred_cy, pred_w, pred_h = _xyxy_to_cxcywh(pred_boxes_xyxy, eps=eps)
    target_cx, target_cy, target_w, target_h = _xyxy_to_cxcywh(target_boxes_xyxy, eps=eps)

    center_sq = (pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)
    size_sq = ((pred_w - target_w).pow(2) + (pred_h - target_h).pow(2)) * 0.25
    wasserstein = torch.sqrt((center_sq + size_sq).clamp_min(eps))
    return torch.exp(-wasserstein / scale)


def wniou_loss_from_boxes(
    pred_boxes_xyxy: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    *,
    lambda_nwd: float = 0.30,
    focus_alpha: float = 0.50,
    focus_gamma: float = 1.00,
    distance_scale: float = 0.05,
    eps: float = 1e-9,
    ciou_fn: Any | None = None,
) -> torch.Tensor:
    """Compute a weighted CIoU+NWD hybrid loss for aligned box pairs."""
    if ciou_fn is None:
        raise ValueError("ciou_fn must be provided to compute the CIoU term.")

    ciou = ciou_fn(pred_boxes_xyxy, target_boxes_xyxy, xywh=False, CIoU=True)
    if ciou.ndim > 1:
        ciou = ciou.squeeze(-1)
    ciou = ciou.to(dtype=pred_boxes_xyxy.dtype)
    nwd = normalized_wasserstein_similarity(
        pred_boxes_xyxy=pred_boxes_xyxy,
        target_boxes_xyxy=target_boxes_xyxy,
        distance_scale=distance_scale,
        eps=eps,
    ).to(dtype=pred_boxes_xyxy.dtype)

    lambda_nwd = float(max(0.0, min(1.0, lambda_nwd)))
    base_loss = (1.0 - lambda_nwd) * (1.0 - ciou) + lambda_nwd * (1.0 - nwd)
    difficulty = (1.0 - ciou.detach().clamp(max=1.0)).clamp_min(0.0).pow(float(max(focus_gamma, 0.0)))
    focus = 1.0 + float(max(focus_alpha, 0.0)) * difficulty
    return base_loss.clamp_min(0.0) * focus


def apply_wniou_patch(
    *,
    lambda_nwd: float = 0.30,
    focus_alpha: float = 0.50,
    focus_gamma: float = 1.00,
    distance_scale: float = 0.05,
) -> dict[str, float]:
    """Patch Ultralytics BboxLoss.forward to use a WNIoU-style hybrid loss."""
    import ultralytics.utils.loss as ultralytics_loss

    bbox_loss_cls = ultralytics_loss.BboxLoss
    original_forward = getattr(bbox_loss_cls, "_wood_q2_original_forward", bbox_loss_cls.forward)
    patch_config = {
        "lambda_nwd": float(lambda_nwd),
        "focus_alpha": float(focus_alpha),
        "focus_gamma": float(focus_gamma),
        "distance_scale": float(distance_scale),
    }

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        target_scores_sum_safe = target_scores_sum.clamp_min(1.0) if torch.is_tensor(target_scores_sum) else max(float(target_scores_sum), 1.0)
        if fg_mask.any():
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            loss_iou = wniou_loss_from_boxes(
                pred_boxes_xyxy=pred_bboxes[fg_mask],
                target_boxes_xyxy=target_bboxes[fg_mask],
                lambda_nwd=patch_config["lambda_nwd"],
                focus_alpha=patch_config["focus_alpha"],
                focus_gamma=patch_config["focus_gamma"],
                distance_scale=patch_config["distance_scale"],
                ciou_fn=ultralytics_loss.bbox_iou,
            )
            loss_iou = (loss_iou.unsqueeze(-1) * weight).sum() / target_scores_sum_safe
        else:
            weight = pred_dist.new_zeros((0, 1))
            loss_iou = pred_dist.new_tensor(0.0)

        if getattr(self, "dfl_loss", None):
            if fg_mask.any():
                target_ltrb = ultralytics_loss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                loss_dfl = (loss_dfl * weight).sum() / target_scores_sum_safe
            else:
                loss_dfl = pred_dist.new_tensor(0.0)
        else:
            loss_dfl = pred_dist.new_tensor(0.0)
        return loss_iou, loss_dfl

    bbox_loss_cls.forward = forward
    bbox_loss_cls._wood_q2_original_forward = original_forward
    bbox_loss_cls._wood_q2_wniou_config = patch_config
    return patch_config
