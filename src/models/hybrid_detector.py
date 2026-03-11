"""Hybrid CNN-Transformer detector with optional P2 branch for ablations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.models.backbones.cnn_backbone import CNNBackbone
from src.models.backbones.transformer_block import SimpleTransformerBlock
from src.models.heads.detection_head import DetectionHead
from src.models.necks.light_neck import LightNeck


def _normalize_transformer_levels(levels: Sequence[str] | None, num_blocks: int) -> list[str]:
    default_levels = ["p4", "p5"]
    candidate_levels = list(levels or default_levels)
    candidate_levels = [level for level in candidate_levels if level in {"p4", "p5"}]
    if num_blocks < 0 or num_blocks > 2:
        raise ValueError("HybridDetector supports 0, 1, or 2 transformer blocks.")
    return candidate_levels[:num_blocks]


class HybridDetector(nn.Module):
    """CNN-dominant hybrid detector that keeps ablations explicit and small."""

    def __init__(
        self,
        num_classes: int = 10,
        use_transformer: bool = True,
        num_transformer_blocks: int = 2,
        use_p2_branch: bool = True,
        neck_out_channels: int = 128,
        transformer_heads: int = 4,
        transformer_levels: Sequence[str] | None = None,
        stage_channels: Sequence[int] = (48, 96, 160, 224),
        stage_depths: Sequence[int] = (1, 2, 2, 2),
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        box_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        objectness_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if use_transformer and num_transformer_blocks not in {1, 2}:
            raise ValueError("Set num_transformer_blocks to 1 or 2 when use_transformer is enabled.")
        if not use_transformer:
            num_transformer_blocks = 0

        self.backbone = CNNBackbone(
            stage_channels=stage_channels,
            stage_depths=stage_depths,
        )
        self.use_transformer = bool(use_transformer)
        self.use_p2_branch = bool(use_p2_branch)
        self.num_classes = int(num_classes)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_detections = int(max_detections)
        self.box_loss_weight = float(box_loss_weight)
        self.cls_loss_weight = float(cls_loss_weight)
        self.objectness_loss_weight = float(objectness_loss_weight)
        self.transformer_levels = _normalize_transformer_levels(
            levels=transformer_levels,
            num_blocks=num_transformer_blocks,
        )
        self.transformer_blocks = nn.ModuleDict(
            {
                level_name: SimpleTransformerBlock(
                    dim=self.backbone.out_channels[level_name],
                    num_heads=transformer_heads,
                )
                for level_name in self.transformer_levels
            }
        )
        self.neck = LightNeck(
            in_channels=self.backbone.out_channels,
            out_channels=neck_out_channels,
            use_p2_branch=self.use_p2_branch,
        )
        self.head = DetectionHead(
            num_classes=num_classes,
            in_channels=neck_out_channels,
        )

    def get_variant_name(self) -> str:
        """Return a short ablation label for logging and debugging."""
        parts = ["cnn"]
        if self.use_transformer:
            parts.append(f"transformer{len(self.transformer_levels)}")
        if self.use_p2_branch:
            parts.append("p2")
        return "+".join(parts)

    def _forward_dense(self, x: torch.Tensor) -> Dict[str, Any]:
        """Return backbone features, neck features, and dense prediction tensors."""
        backbone_features = self.backbone(x)
        refined_features = dict(backbone_features)

        for level_name in self.transformer_levels:
            refined_features[level_name] = self.transformer_blocks[level_name](refined_features[level_name])

        pyramid_features = self.neck(refined_features)
        predictions = self.head(pyramid_features)
        predictions.update(
            {
                "backbone_features": refined_features,
                "neck_features": pyramid_features,
                "feature_levels": list(pyramid_features.keys()),
                "variant_name": self.get_variant_name(),
            }
        )
        return predictions

    def _stack_images(self, images: Sequence[torch.Tensor]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Pad a list of images into a single batch tensor."""
        image_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
        max_height = max(height for height, _ in image_sizes)
        max_width = max(width for _, width in image_sizes)
        batch = images[0].new_zeros((len(images), images[0].shape[0], max_height, max_width))
        for index, image in enumerate(images):
            height, width = image_sizes[index]
            batch[index, :, :height, :width] = image
        return batch, image_sizes

    def _resolve_feature_strides(
        self,
        dense_outputs: Mapping[str, Any],
        image_height: int,
        image_width: int,
    ) -> Dict[str, tuple[float, float]]:
        strides: Dict[str, tuple[float, float]] = {}
        for level_name, feature in dense_outputs["objectness"].items():
            stride_y = float(image_height) / float(feature.shape[-2])
            stride_x = float(image_width) / float(feature.shape[-1])
            strides[level_name] = (stride_y, stride_x)
        return strides

    def _select_target_level(self, box: torch.Tensor, level_names: Sequence[str]) -> str:
        width = float((box[2] - box[0]).clamp(min=1.0).item())
        height = float((box[3] - box[1]).clamp(min=1.0).item())
        object_scale = max(width, height)
        if "p2" in level_names and object_scale <= 64.0:
            return "p2"
        if object_scale <= 128.0:
            return "p3"
        if object_scale <= 256.0:
            return "p4"
        return "p5"

    def _build_training_targets(
        self,
        dense_outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_sizes: Sequence[tuple[int, int]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        level_names = list(dense_outputs["feature_levels"])
        training_targets: Dict[str, Dict[str, torch.Tensor]] = {}
        for level_name in level_names:
            logits = dense_outputs["cls_logits"][level_name]
            batch_size, _, height, width = logits.shape
            training_targets[level_name] = {
                "objectness": torch.zeros((batch_size, 1, height, width), device=logits.device),
                "classes": torch.zeros((batch_size, self.num_classes, height, width), device=logits.device),
                "boxes": torch.zeros((batch_size, 4, height, width), device=logits.device),
                "positive_mask": torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device=logits.device),
            }

        for batch_index, (target, image_size) in enumerate(zip(targets, image_sizes)):
            image_height, image_width = image_size
            feature_strides = self._resolve_feature_strides(
                dense_outputs=dense_outputs,
                image_height=image_height,
                image_width=image_width,
            )
            boxes = target["boxes"]
            labels = target["labels"] - 1
            for box, label in zip(boxes, labels):
                if int(label) < 0 or int(label) >= self.num_classes:
                    continue
                level_name = self._select_target_level(box, level_names)
                stride_y, stride_x = feature_strides[level_name]
                target_blob = training_targets[level_name]

                center_x = float((box[0] + box[2]) * 0.5)
                center_y = float((box[1] + box[3]) * 0.5)
                grid_x = int(min(max(center_x / stride_x, 0.0), target_blob["objectness"].shape[-1] - 1))
                grid_y = int(min(max(center_y / stride_y, 0.0), target_blob["objectness"].shape[-2] - 1))

                cell_center_x = (grid_x + 0.5) * stride_x
                cell_center_y = (grid_y + 0.5) * stride_y
                box_target = torch.tensor(
                    [
                        max(cell_center_x - float(box[0]), 0.0),
                        max(cell_center_y - float(box[1]), 0.0),
                        max(float(box[2]) - cell_center_x, 0.0),
                        max(float(box[3]) - cell_center_y, 0.0),
                    ],
                    dtype=torch.float32,
                    device=boxes.device,
                )

                target_blob["objectness"][batch_index, 0, grid_y, grid_x] = 1.0
                target_blob["classes"][batch_index, int(label), grid_y, grid_x] = 1.0
                target_blob["boxes"][batch_index, :, grid_y, grid_x] = box_target
                target_blob["positive_mask"][batch_index, 0, grid_y, grid_x] = True
        return training_targets

    def _compute_losses(
        self,
        dense_outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_sizes: Sequence[tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        training_targets = self._build_training_targets(
            dense_outputs=dense_outputs,
            targets=targets,
            image_sizes=image_sizes,
        )

        cls_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        box_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        objectness_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for level_name in dense_outputs["feature_levels"]:
            objectness_logits = dense_outputs["objectness"][level_name]
            cls_logits = dense_outputs["cls_logits"][level_name]
            box_regression = dense_outputs["bbox_regression"][level_name]
            target_blob = training_targets[level_name]
            positive_mask = target_blob["positive_mask"]

            objectness_loss = objectness_loss + F.binary_cross_entropy_with_logits(
                objectness_logits,
                target_blob["objectness"],
            )

            if positive_mask.any():
                expanded_positive_mask = positive_mask.expand_as(cls_logits)
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    cls_logits[expanded_positive_mask],
                    target_blob["classes"][expanded_positive_mask],
                )
                box_positive_mask = positive_mask.expand_as(box_regression)
                box_loss = box_loss + F.l1_loss(
                    box_regression[box_positive_mask],
                    target_blob["boxes"][box_positive_mask],
                )

        return {
            "loss_objectness": objectness_loss * self.objectness_loss_weight,
            "loss_cls": cls_loss * self.cls_loss_weight,
            "loss_box_reg": box_loss * self.box_loss_weight,
        }

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        order = scores.argsort(descending=True)
        keep: list[int] = []

        while order.numel() > 0:
            current = int(order[0])
            keep.append(current)
            if order.numel() == 1:
                break

            remaining = order[1:]
            xx1 = torch.maximum(x1[current], x1[remaining])
            yy1 = torch.maximum(y1[current], y1[remaining])
            xx2 = torch.minimum(x2[current], x2[remaining])
            yy2 = torch.minimum(y2[current], y2[remaining])

            inter_w = (xx2 - xx1).clamp(min=0)
            inter_h = (yy2 - yy1).clamp(min=0)
            intersection = inter_w * inter_h
            union = areas[current] + areas[remaining] - intersection
            iou = intersection / union.clamp(min=1e-6)
            order = remaining[iou <= iou_threshold]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _decode_predictions(
        self,
        dense_outputs: Mapping[str, Any],
        image_sizes: Sequence[tuple[int, int]],
    ) -> list[Dict[str, torch.Tensor]]:
        detections: list[Dict[str, torch.Tensor]] = []
        level_names = list(dense_outputs["feature_levels"])

        for batch_index, image_size in enumerate(image_sizes):
            image_height, image_width = image_size
            feature_strides = self._resolve_feature_strides(
                dense_outputs=dense_outputs,
                image_height=image_height,
                image_width=image_width,
            )
            image_boxes = []
            image_scores = []
            image_labels = []

            for level_name in level_names:
                cls_logits = dense_outputs["cls_logits"][level_name][batch_index]
                objectness_logits = dense_outputs["objectness"][level_name][batch_index]
                box_regression = dense_outputs["bbox_regression"][level_name][batch_index]
                _, height, width = objectness_logits.shape
                stride_y, stride_x = feature_strides[level_name]

                cls_scores = torch.sigmoid(cls_logits)
                obj_scores = torch.sigmoid(objectness_logits[0])
                combined_scores = cls_scores * obj_scores.unsqueeze(0)
                class_scores, class_ids = combined_scores.max(dim=0)
                keep_mask = class_scores >= self.score_threshold
                if not keep_mask.any():
                    continue

                kept_scores = class_scores[keep_mask]
                kept_labels = class_ids[keep_mask] + 1
                kept_positions = keep_mask.nonzero(as_tuple=False)
                kept_boxes = []
                for grid_y, grid_x in kept_positions:
                    l, t, r, b = box_regression[:, grid_y, grid_x]
                    center_x = (float(grid_x) + 0.5) * stride_x
                    center_y = (float(grid_y) + 0.5) * stride_y
                    box = torch.tensor(
                        [
                            max(center_x - float(l), 0.0),
                            max(center_y - float(t), 0.0),
                            min(center_x + float(r), float(image_width)),
                            min(center_y + float(b), float(image_height)),
                        ],
                        dtype=torch.float32,
                        device=box_regression.device,
                    )
                    kept_boxes.append(box)

                if kept_boxes:
                    image_boxes.append(torch.stack(kept_boxes, dim=0))
                    image_scores.append(kept_scores)
                    image_labels.append(kept_labels)

            if image_boxes:
                boxes = torch.cat(image_boxes, dim=0)
                scores = torch.cat(image_scores, dim=0)
                labels = torch.cat(image_labels, dim=0)
                keep = self._nms(boxes, scores, self.nms_threshold)[: self.max_detections]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            detections.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
            )
        return detections

    def forward(
        self,
        images: torch.Tensor | Sequence[torch.Tensor],
        targets: Sequence[Mapping[str, torch.Tensor]] | None = None,
    ) -> Dict[str, Any] | list[Dict[str, torch.Tensor]]:
        """Support dense feature inspection, training losses, and inference detections."""
        if torch.is_tensor(images):
            return self._forward_dense(images)

        batch_tensor, image_sizes = self._stack_images(images)
        dense_outputs = self._forward_dense(batch_tensor)
        if targets is not None:
            return self._compute_losses(
                dense_outputs=dense_outputs,
                targets=targets,
                image_sizes=image_sizes,
            )
        return self._decode_predictions(
            dense_outputs=dense_outputs,
            image_sizes=image_sizes,
        )


def build_hybrid_detector(model_config: Dict[str, Any], train_config: Dict[str, Any] | None = None) -> HybridDetector:
    """Factory wrapper so hybrid ablations can be config-driven later."""
    _ = train_config
    return HybridDetector(
        num_classes=int(model_config.get("num_classes", 10)),
        use_transformer=bool(model_config.get("use_transformer", True)),
        num_transformer_blocks=int(model_config.get("num_transformer_blocks", 2)),
        use_p2_branch=bool(model_config.get("use_p2_branch", True)),
        neck_out_channels=int(model_config.get("neck_out_channels", 128)),
        transformer_heads=int(model_config.get("transformer_heads", 4)),
        transformer_levels=model_config.get("transformer_levels"),
        stage_channels=tuple(model_config.get("stage_channels", (48, 96, 160, 224))),
        stage_depths=tuple(model_config.get("stage_depths", (1, 2, 2, 2))),
        score_threshold=float(model_config.get("score_threshold", 0.05)),
        nms_threshold=float(model_config.get("nms_threshold", 0.5)),
        max_detections=int(model_config.get("max_detections", 100)),
        box_loss_weight=float(model_config.get("box_loss_weight", 1.0)),
        cls_loss_weight=float(model_config.get("cls_loss_weight", 1.0)),
        objectness_loss_weight=float(model_config.get("objectness_loss_weight", 1.0)),
    )
