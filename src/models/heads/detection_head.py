"""Baseline detector factory and lightweight dense head for hybrid ablations."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class _HeadConvBlock(nn.Sequential):
    """Small conv block for the lightweight dense prediction head."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )


class DetectionHead(nn.Module):
    """Compact per-level dense head for hybrid feature ablations."""

    def __init__(self, num_classes: int = 10, in_channels: int = 128, num_head_convs: int = 2) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        conv_blocks = [_HeadConvBlock(self.in_channels) for _ in range(max(int(num_head_convs), 1))]
        self.shared_tower = nn.Sequential(*conv_blocks)
        self.classification_head = nn.Conv2d(
            self.in_channels,
            self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.box_regression_head = nn.Conv2d(
            self.in_channels,
            4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.objectness_head = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, features):
        """Predict dense logits and box deltas for each pyramid level."""
        cls_logits = {}
        bbox_regression = {}
        objectness = {}
        refined_features = {}

        for level_name, feature in features.items():
            refined_feature = self.shared_tower(feature)
            refined_features[level_name] = refined_feature
            cls_logits[level_name] = self.classification_head(refined_feature)
            bbox_regression[level_name] = torch.relu(self.box_regression_head(refined_feature))
            objectness[level_name] = self.objectness_head(refined_feature)

        return {
            "features": refined_features,
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "objectness": objectness,
        }


def build_baseline_detector(model_config: Dict[str, Any], train_config: Dict[str, Any] | None = None) -> nn.Module:
    """Build a lightweight baseline detector using torchvision detection models."""
    try:
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        try:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
        except ImportError:  # pragma: no cover - torchvision version dependent
            fasterrcnn_mobilenet_v3_large_320_fpn = None
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "torchvision is required for the baseline detector. Install the dependencies "
            "from requirements.txt or environment.yml before training or evaluation."
        ) from exc

    train_config = train_config or {}
    num_classes = int(model_config.get("num_classes", 1))
    image_size = int(model_config.get("image_size", train_config.get("image_size", 1024)))
    backbone_name = str(model_config.get("backbone", "mobilenet")).lower()

    detector = None
    if backbone_name == "mobilenet" and fasterrcnn_mobilenet_v3_large_320_fpn is not None:
        try:
            detector = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None,
                weights_backbone=None,
                min_size=image_size,
                max_size=image_size,
            )
        except TypeError:  # pragma: no cover - torchvision version dependent
            detector = fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=False,
                pretrained_backbone=False,
                min_size=image_size,
                max_size=image_size,
            )
    else:
        try:
            detector = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                min_size=image_size,
                max_size=image_size,
            )
        except TypeError:  # pragma: no cover - torchvision version dependent
            detector = fasterrcnn_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=False,
                min_size=image_size,
                max_size=image_size,
            )

    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    detector.roi_heads.detections_per_img = int(model_config.get("max_detections", 100))
    detector.roi_heads.score_thresh = float(model_config.get("score_threshold", 0.05))
    detector.roi_heads.nms_thresh = float(model_config.get("nms_threshold", 0.5))
    return detector
