"""Baseline detector factory and lightweight dense head for hybrid ablations."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn


class _HeadConvBlock(nn.Sequential):
    """Depthwise-separable head block for lightweight dense prediction."""

    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )


class DetectionHead(nn.Module):
    """Compact FCOS-like dense head for hybrid feature ablations."""

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 128,
        num_head_convs: int = 2,
        classification_prior: float = 0.01,
        centerness_prior: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.num_head_convs = max(int(num_head_convs), 1)

        self.classification_tower = nn.Sequential(
            *[_HeadConvBlock(self.in_channels) for _ in range(self.num_head_convs)]
        )
        self.regression_tower = nn.Sequential(
            *[_HeadConvBlock(self.in_channels) for _ in range(self.num_head_convs)]
        )
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
        self.centerness_head = nn.Conv2d(
            self.in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._init_parameters(
            classification_prior=classification_prior,
            centerness_prior=centerness_prior,
        )

    def _init_parameters(self, classification_prior: float, centerness_prior: float) -> None:
        modules = [
            self.classification_tower,
            self.regression_tower,
            self.classification_head,
            self.box_regression_head,
            self.centerness_head,
        ]
        for module in modules:
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    nn.init.normal_(child.weight, std=0.01)
                    if child.bias is not None:
                        nn.init.constant_(child.bias, 0.0)

        prior_bias = -math.log((1.0 - classification_prior) / classification_prior)
        centerness_bias = -math.log((1.0 - centerness_prior) / centerness_prior)
        nn.init.constant_(self.classification_head.bias, prior_bias)
        nn.init.constant_(self.centerness_head.bias, centerness_bias)

    def forward(self, features: Mapping[str, torch.Tensor]):
        """Predict dense logits, centerness, and box deltas for each pyramid level."""
        cls_logits = {}
        bbox_regression = {}
        centerness = {}
        refined_features = {}

        for level_name, feature in features.items():
            cls_feature = self.classification_tower(feature)
            reg_feature = self.regression_tower(feature)
            refined_features[level_name] = reg_feature
            cls_logits[level_name] = self.classification_head(cls_feature)
            bbox_regression[level_name] = F.softplus(self.box_regression_head(reg_feature))
            centerness[level_name] = self.centerness_head(reg_feature)

        return {
            "features": refined_features,
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "centerness": centerness,
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
