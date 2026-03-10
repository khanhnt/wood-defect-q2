"""Baseline detection model factory and placeholder head for later ablations."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn


class DetectionHead(nn.Module):
    """Placeholder head retained for the unfinished hybrid model path."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Identity()

    def forward(self, features):
        """Forward pass."""
        return self.head(features)


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
