"""Baseline detector factory and lightweight dense head for hybrid ablations."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbones.torchvision_fpn import build_torchvision_fpn_backbone


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
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        try:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        except ImportError:  # pragma: no cover - torchvision version dependent
            fasterrcnn_mobilenet_v3_large_fpn = None
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
    trainable_backbone_layers = model_config.get("trainable_backbone_layers")
    if trainable_backbone_layers is not None:
        trainable_backbone_layers = int(trainable_backbone_layers)
    small_defect_profile = str(model_config.get("small_defect_profile", "none")).lower()

    detector_kwargs: Dict[str, Any] = {
        "min_size": image_size,
        "max_size": image_size,
    }
    if trainable_backbone_layers is not None:
        detector_kwargs["trainable_backbone_layers"] = trainable_backbone_layers

    if small_defect_profile not in {"none", "small"}:
        raise ValueError(f"Unsupported small_defect_profile={small_defect_profile!r}.")

    detector = None
    if backbone_name in {"mobilenet", "mobilenet_320"}:
        if fasterrcnn_mobilenet_v3_large_320_fpn is None:
            raise ImportError("This torchvision build does not provide fasterrcnn_mobilenet_v3_large_320_fpn.")
        try:
            detector = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None,
                weights_backbone=None,
                **detector_kwargs,
            )
        except TypeError:  # pragma: no cover - torchvision version dependent
            detector = fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=False,
                pretrained_backbone=False,
                **detector_kwargs,
            )
    elif backbone_name in {"mobilenet_hr", "mobilenet_fpn"}:
        if fasterrcnn_mobilenet_v3_large_fpn is None:
            raise ImportError("This torchvision build does not provide fasterrcnn_mobilenet_v3_large_fpn.")
        try:
            detector = fasterrcnn_mobilenet_v3_large_fpn(
                weights=None,
                weights_backbone=None,
                **detector_kwargs,
            )
        except TypeError:  # pragma: no cover - torchvision version dependent
            detector = fasterrcnn_mobilenet_v3_large_fpn(
                pretrained=False,
                pretrained_backbone=False,
                **detector_kwargs,
            )
    elif backbone_name in {"densenet", "densenet121", "maxvit", "maxvit_t"}:
        backbone = build_torchvision_fpn_backbone(
            backbone_name=backbone_name,
            out_channels=int(model_config.get("fpn_out_channels", 256)),
        )
        detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes + 1,
            min_size=image_size,
            max_size=image_size,
        )
    else:
        try:
            detector = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                **detector_kwargs,
            )
        except TypeError:  # pragma: no cover - torchvision version dependent
            detector = fasterrcnn_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=False,
                **detector_kwargs,
            )

    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    detector.roi_heads.detections_per_img = int(model_config.get("max_detections", 100))
    detector.roi_heads.score_thresh = float(model_config.get("score_threshold", 0.05))
    detector.roi_heads.nms_thresh = float(model_config.get("nms_threshold", 0.5))
    if small_defect_profile == "small":
        num_feature_maps = len(detector.rpn.anchor_generator.sizes)
        anchor_sizes = tuple((int(8 * (2**level_index)),) for level_index in range(num_feature_maps))
        aspect_ratios = tuple((0.5, 1.0, 2.0) for _ in range(num_feature_maps))
        detector.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        detector.rpn._pre_nms_top_n["training"] = int(model_config.get("rpn_pre_nms_top_n_train", 3000))
        detector.rpn._pre_nms_top_n["testing"] = int(model_config.get("rpn_pre_nms_top_n_test", 2000))
        detector.rpn._post_nms_top_n["training"] = int(model_config.get("rpn_post_nms_top_n_train", 1500))
        detector.rpn._post_nms_top_n["testing"] = int(model_config.get("rpn_post_nms_top_n_test", 1000))
    return detector
