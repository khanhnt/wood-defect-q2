from __future__ import annotations

import torch

from src.losses.yolo_wniou import normalized_wasserstein_similarity, wniou_loss_from_boxes


def _mock_ciou(pred_boxes, target_boxes, xywh=False, CIoU=True):
    del xywh, CIoU
    diff = torch.abs(pred_boxes - target_boxes).mean(dim=-1, keepdim=True)
    return (1.0 - diff).clamp(min=-1.0, max=1.0)


def test_normalized_wasserstein_similarity_is_higher_for_closer_boxes():
    target = torch.tensor([[0.20, 0.20, 0.40, 0.40]], dtype=torch.float32)
    near = torch.tensor([[0.21, 0.20, 0.41, 0.40]], dtype=torch.float32)
    far = torch.tensor([[0.40, 0.40, 0.70, 0.70]], dtype=torch.float32)

    near_score = normalized_wasserstein_similarity(near, target)
    far_score = normalized_wasserstein_similarity(far, target)

    assert torch.all(near_score > far_score)


def test_wniou_loss_is_higher_for_harder_boxes():
    target = torch.tensor([[0.20, 0.20, 0.40, 0.40]], dtype=torch.float32)
    easier = torch.tensor([[0.21, 0.20, 0.41, 0.40]], dtype=torch.float32)
    harder = torch.tensor([[0.40, 0.40, 0.70, 0.70]], dtype=torch.float32)

    easier_loss = wniou_loss_from_boxes(easier, target, ciou_fn=_mock_ciou)
    harder_loss = wniou_loss_from_boxes(harder, target, ciou_fn=_mock_ciou)

    assert torch.all(harder_loss > easier_loss)


def test_wniou_loss_respects_nwd_weight_bounds():
    pred = torch.tensor([[0.20, 0.20, 0.40, 0.40]], dtype=torch.float32)
    target = torch.tensor([[0.20, 0.20, 0.40, 0.40]], dtype=torch.float32)

    loss = wniou_loss_from_boxes(pred, target, lambda_nwd=1.5, ciou_fn=_mock_ciou)

    assert torch.isfinite(loss).all()
    assert torch.all(loss >= 0.0)
