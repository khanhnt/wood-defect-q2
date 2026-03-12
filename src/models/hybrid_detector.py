"""Hybrid CNN-Transformer detector with FCOS-like dense detection for ablations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.detection_loss import generalized_box_iou_loss, sigmoid_focal_loss
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
    """CNN-dominant hybrid detector with a lightweight FCOS-style head."""

    def __init__(
        self,
        num_classes: int = 10,
        use_transformer: bool = True,
        num_transformer_blocks: int = 2,
        use_p2_branch: bool = True,
        neck_out_channels: int = 128,
        transformer_heads: int = 4,
        transformer_levels: Sequence[str] | None = None,
        stage_channels: Sequence[int] = (64, 128, 192, 256),
        stage_depths: Sequence[int] = (1, 2, 3, 2),
        score_threshold: float = 0.05,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        pre_nms_topk: int = 1000,
        center_sampling_radius: float = 1.5,
        box_loss_weight: float = 2.0,
        cls_loss_weight: float = 1.0,
        centerness_loss_weight: float = 1.0,
        num_head_convs: int = 2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        classification_prior: float = 0.01,
        centerness_prior: float = 0.01,
        normalize_inputs: bool = True,
        input_mean: Sequence[float] = (0.485, 0.456, 0.406),
        input_std: Sequence[float] = (0.229, 0.224, 0.225),
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
        self.pre_nms_topk = int(pre_nms_topk)
        self.center_sampling_radius = float(center_sampling_radius)
        self.box_loss_weight = float(box_loss_weight)
        self.cls_loss_weight = float(cls_loss_weight)
        self.centerness_loss_weight = float(centerness_loss_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.normalize_inputs = bool(normalize_inputs)

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
            num_head_convs=num_head_convs,
            classification_prior=classification_prior,
            centerness_prior=centerness_prior,
        )
        mean_tensor = torch.tensor(list(input_mean), dtype=torch.float32).view(1, -1, 1, 1)
        std_tensor = torch.tensor(list(input_std), dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("input_mean", mean_tensor, persistent=False)
        self.register_buffer("input_std", std_tensor, persistent=False)

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
        if self.normalize_inputs:
            x = (x - self.input_mean) / self.input_std.clamp(min=1e-6)
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
        for level_name, feature in dense_outputs["centerness"].items():
            stride_y = float(image_height) / float(feature.shape[-2])
            stride_x = float(image_width) / float(feature.shape[-1])
            strides[level_name] = (stride_y, stride_x)
        return strides

    def _build_regression_ranges(self, level_names: Sequence[str]) -> Dict[str, tuple[float, float]]:
        if "p2" in level_names:
            defaults = {
                "p2": (0.0, 64.0),
                "p3": (48.0, 128.0),
                "p4": (96.0, 256.0),
                "p5": (192.0, float("inf")),
            }
        else:
            defaults = {
                "p3": (0.0, 128.0),
                "p4": (96.0, 256.0),
                "p5": (192.0, float("inf")),
            }
        return {level_name: defaults[level_name] for level_name in level_names}

    def _compute_level_centers(
        self,
        height: int,
        width: int,
        stride_y: float,
        stride_x: float,
        device: torch.device,
    ) -> torch.Tensor:
        grid_y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride_y
        grid_x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride_x
        center_y, center_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        return torch.stack((center_x.reshape(-1), center_y.reshape(-1)), dim=-1)

    def _encode_targets_for_level(
        self,
        centers: torch.Tensor,
        level_name: str,
        strides: tuple[float, float],
        boxes: torch.Tensor,
        labels: torch.Tensor,
        regression_ranges: Mapping[str, tuple[float, float]],
    ) -> Dict[str, torch.Tensor]:
        num_points = centers.shape[0]
        device = centers.device
        labels_out = torch.full((num_points,), -1, dtype=torch.long, device=device)
        bbox_targets = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        target_boxes = torch.zeros((num_points, 4), dtype=torch.float32, device=device)
        centerness_targets = torch.zeros((num_points,), dtype=torch.float32, device=device)

        if boxes.numel() == 0:
            return {
                "labels": labels_out,
                "bbox_targets": bbox_targets,
                "target_boxes": target_boxes,
                "centerness_targets": centerness_targets,
            }

        center_x = centers[:, 0].unsqueeze(1)
        center_y = centers[:, 1].unsqueeze(1)
        x1 = boxes[:, 0].unsqueeze(0)
        y1 = boxes[:, 1].unsqueeze(0)
        x2 = boxes[:, 2].unsqueeze(0)
        y2 = boxes[:, 3].unsqueeze(0)

        l = center_x - x1
        t = center_y - y1
        r = x2 - center_x
        b = y2 - center_y
        ltrb = torch.stack((l, t, r, b), dim=-1)

        inside_box = ltrb.min(dim=-1).values >= 0.0
        max_regression = ltrb.max(dim=-1).values
        lower_bound, upper_bound = regression_ranges[level_name]
        in_range = max_regression >= lower_bound
        if upper_bound != float("inf"):
            in_range = in_range & (max_regression <= upper_bound)

        gt_center_x = (x1 + x2) * 0.5
        gt_center_y = (y1 + y2) * 0.5
        sampling_radius = self.center_sampling_radius * max(strides)
        sample_x1 = torch.maximum(gt_center_x - sampling_radius, x1)
        sample_y1 = torch.maximum(gt_center_y - sampling_radius, y1)
        sample_x2 = torch.minimum(gt_center_x + sampling_radius, x2)
        sample_y2 = torch.minimum(gt_center_y + sampling_radius, y2)
        inside_center = (
            (center_x >= sample_x1)
            & (center_x <= sample_x2)
            & (center_y >= sample_y1)
            & (center_y <= sample_y2)
        )

        valid_locations = inside_box & in_range & inside_center
        missing_gt_mask = ~valid_locations.any(dim=0)
        if missing_gt_mask.any():
            fallback_locations = inside_box & in_range
            for gt_index in missing_gt_mask.nonzero(as_tuple=False).squeeze(1):
                if fallback_locations[:, gt_index].any():
                    valid_locations[:, gt_index] = fallback_locations[:, gt_index]

        gt_areas = ((x2 - x1) * (y2 - y1)).expand(num_points, -1).clone()
        gt_areas[~valid_locations] = float("inf")
        min_areas, matched_indices = gt_areas.min(dim=1)
        positive_mask = torch.isfinite(min_areas)

        if not positive_mask.any():
            return {
                "labels": labels_out,
                "bbox_targets": bbox_targets,
                "target_boxes": target_boxes,
                "centerness_targets": centerness_targets,
            }

        positive_indices = positive_mask.nonzero(as_tuple=False).squeeze(1)
        matched_targets = matched_indices[positive_indices]
        matched_boxes = boxes[matched_targets]
        matched_labels = labels[matched_targets]
        matched_ltrb = ltrb[positive_indices, matched_targets]

        left_right = matched_ltrb[:, [0, 2]]
        top_bottom = matched_ltrb[:, [1, 3]]
        centerness = torch.sqrt(
            (
                left_right.min(dim=-1).values
                / left_right.max(dim=-1).values.clamp(min=1e-6)
            )
            * (
                top_bottom.min(dim=-1).values
                / top_bottom.max(dim=-1).values.clamp(min=1e-6)
            )
        )

        labels_out[positive_indices] = matched_labels
        bbox_targets[positive_indices] = matched_ltrb
        target_boxes[positive_indices] = matched_boxes
        centerness_targets[positive_indices] = centerness
        return {
            "labels": labels_out,
            "bbox_targets": bbox_targets,
            "target_boxes": target_boxes,
            "centerness_targets": centerness_targets,
        }

    def _prepare_level_targets(
        self,
        dense_outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_sizes: Sequence[tuple[int, int]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        level_names = list(dense_outputs["feature_levels"])
        regression_ranges = self._build_regression_ranges(level_names)
        target_bundle: Dict[str, Dict[str, torch.Tensor]] = {}

        for level_name in level_names:
            cls_logits = dense_outputs["cls_logits"][level_name]
            batch_size, _, height, width = cls_logits.shape
            labels_per_image = []
            bbox_per_image = []
            target_boxes_per_image = []
            centerness_per_image = []
            centers_per_image = []

            for image_index in range(batch_size):
                image_height, image_width = image_sizes[image_index]
                stride_y = float(image_height) / float(height)
                stride_x = float(image_width) / float(width)
                centers = self._compute_level_centers(
                    height=height,
                    width=width,
                    stride_y=stride_y,
                    stride_x=stride_x,
                    device=cls_logits.device,
                )
                encoded_targets = self._encode_targets_for_level(
                    centers=centers,
                    level_name=level_name,
                    strides=(stride_y, stride_x),
                    boxes=targets[image_index]["boxes"],
                    labels=targets[image_index]["labels"] - 1,
                    regression_ranges=regression_ranges,
                )
                labels_per_image.append(encoded_targets["labels"])
                bbox_per_image.append(encoded_targets["bbox_targets"])
                target_boxes_per_image.append(encoded_targets["target_boxes"])
                centerness_per_image.append(encoded_targets["centerness_targets"])
                centers_per_image.append(centers)

            target_bundle[level_name] = {
                "labels": torch.stack(labels_per_image, dim=0),
                "bbox_targets": torch.stack(bbox_per_image, dim=0),
                "target_boxes": torch.stack(target_boxes_per_image, dim=0),
                "centerness_targets": torch.stack(centerness_per_image, dim=0),
                "centers": torch.stack(centers_per_image, dim=0),
            }
        return target_bundle

    def _decode_ltrb_to_xyxy(self, centers: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                centers[:, 0] - deltas[:, 0],
                centers[:, 1] - deltas[:, 1],
                centers[:, 0] + deltas[:, 2],
                centers[:, 1] + deltas[:, 3],
            ),
            dim=-1,
        )

    def _compute_losses(
        self,
        dense_outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_sizes: Sequence[tuple[int, int]],
    ) -> Dict[str, torch.Tensor]:
        target_bundle = self._prepare_level_targets(
            dense_outputs=dense_outputs,
            targets=targets,
            image_sizes=image_sizes,
        )

        cls_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        box_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        centerness_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        total_positive = 0
        box_weight_sum = torch.tensor(0.0, device=next(self.parameters()).device)

        for level_name in dense_outputs["feature_levels"]:
            cls_logits = dense_outputs["cls_logits"][level_name].permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            bbox_regression = dense_outputs["bbox_regression"][level_name].permute(0, 2, 3, 1).reshape(-1, 4)
            centerness_logits = dense_outputs["centerness"][level_name].permute(0, 2, 3, 1).reshape(-1)

            level_targets = target_bundle[level_name]
            labels = level_targets["labels"].reshape(-1)
            bbox_targets = level_targets["bbox_targets"].reshape(-1, 4)
            target_boxes = level_targets["target_boxes"].reshape(-1, 4)
            centers = level_targets["centers"].reshape(-1, 2)
            centerness_targets = level_targets["centerness_targets"].reshape(-1)
            positive_mask = labels >= 0
            total_positive += int(positive_mask.sum().item())

            cls_targets = torch.zeros_like(cls_logits)
            if positive_mask.any():
                cls_targets[positive_mask, labels[positive_mask]] = 1.0
            cls_loss = cls_loss + sigmoid_focal_loss(
                cls_logits,
                cls_targets,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="sum",
            )

            if positive_mask.any():
                pred_boxes = self._decode_ltrb_to_xyxy(centers[positive_mask], bbox_regression[positive_mask])
                box_weights = centerness_targets[positive_mask]
                box_losses = generalized_box_iou_loss(
                    pred_boxes,
                    target_boxes[positive_mask],
                    reduction="none",
                )
                box_loss = box_loss + (box_losses * box_weights).sum()
                box_weight_sum = box_weight_sum + box_weights.sum()
                centerness_loss = centerness_loss + F.binary_cross_entropy_with_logits(
                    centerness_logits[positive_mask],
                    centerness_targets[positive_mask],
                    reduction="sum",
                )

        normalizer = max(total_positive, 1)
        box_normalizer = torch.clamp(box_weight_sum, min=1.0)
        return {
            "loss_cls": (cls_loss / normalizer) * self.cls_loss_weight,
            "loss_box_reg": (box_loss / box_normalizer) * self.box_loss_weight,
            "loss_centerness": (centerness_loss / normalizer) * self.centerness_loss_weight,
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

    def _class_wise_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        keep_indices = []
        for class_id in labels.unique(sorted=True):
            class_mask = labels == class_id
            class_indices = class_mask.nonzero(as_tuple=False).squeeze(1)
            class_keep = self._nms(
                boxes[class_indices],
                scores[class_indices],
                self.nms_threshold,
            )
            keep_indices.append(class_indices[class_keep])

        if not keep_indices:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)
        keep = torch.cat(keep_indices, dim=0)
        return keep[scores[keep].argsort(descending=True)]

    def _decode_predictions(
        self,
        dense_outputs: Mapping[str, Any],
        image_sizes: Sequence[tuple[int, int]],
    ) -> list[Dict[str, torch.Tensor]]:
        detections: list[Dict[str, torch.Tensor]] = []
        level_names = list(dense_outputs["feature_levels"])

        for batch_index, image_size in enumerate(image_sizes):
            image_height, image_width = image_size
            image_boxes = []
            image_scores = []
            image_labels = []

            for level_name in level_names:
                cls_logits = dense_outputs["cls_logits"][level_name][batch_index]
                bbox_regression = dense_outputs["bbox_regression"][level_name][batch_index]
                centerness_logits = dense_outputs["centerness"][level_name][batch_index, 0]
                _, height, width = cls_logits.shape
                stride_y = float(image_height) / float(height)
                stride_x = float(image_width) / float(width)
                centers = self._compute_level_centers(
                    height=height,
                    width=width,
                    stride_y=stride_y,
                    stride_x=stride_x,
                    device=cls_logits.device,
                )

                cls_scores = torch.sigmoid(cls_logits).permute(1, 2, 0).reshape(-1, self.num_classes)
                centerness = torch.sigmoid(centerness_logits).reshape(-1, 1)
                combined_scores = cls_scores * centerness
                point_scores, class_indices = combined_scores.max(dim=1)
                candidate_mask = point_scores >= self.score_threshold
                if not candidate_mask.any():
                    continue

                candidate_indices = candidate_mask.nonzero(as_tuple=False).squeeze(1)
                candidate_scores = point_scores[candidate_indices]
                topk = min(self.pre_nms_topk, candidate_scores.numel())
                topk_scores, topk_order = candidate_scores.topk(topk)
                topk_indices = candidate_indices[topk_order]

                reg_values = bbox_regression.permute(1, 2, 0).reshape(-1, 4)[topk_indices]
                decoded_boxes = self._decode_ltrb_to_xyxy(centers[topk_indices], reg_values)
                decoded_boxes[:, 0::2] = decoded_boxes[:, 0::2].clamp(min=0.0, max=float(image_width))
                decoded_boxes[:, 1::2] = decoded_boxes[:, 1::2].clamp(min=0.0, max=float(image_height))

                image_boxes.append(decoded_boxes)
                image_scores.append(topk_scores)
                image_labels.append(class_indices[topk_indices] + 1)

            if image_boxes:
                boxes = torch.cat(image_boxes, dim=0)
                scores = torch.cat(image_scores, dim=0)
                labels = torch.cat(image_labels, dim=0)
                keep = self._class_wise_nms(boxes, scores, labels)[: self.max_detections]
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
        stage_channels=tuple(model_config.get("stage_channels", (64, 128, 192, 256))),
        stage_depths=tuple(model_config.get("stage_depths", (1, 2, 3, 2))),
        score_threshold=float(model_config.get("score_threshold", 0.05)),
        nms_threshold=float(model_config.get("nms_threshold", 0.5)),
        max_detections=int(model_config.get("max_detections", 100)),
        pre_nms_topk=int(model_config.get("pre_nms_topk", 1000)),
        center_sampling_radius=float(model_config.get("center_sampling_radius", 1.5)),
        box_loss_weight=float(model_config.get("box_loss_weight", 2.0)),
        cls_loss_weight=float(model_config.get("cls_loss_weight", 1.0)),
        centerness_loss_weight=float(model_config.get("centerness_loss_weight", 1.0)),
        num_head_convs=int(model_config.get("num_head_convs", 2)),
        focal_alpha=float(model_config.get("focal_alpha", 0.25)),
        focal_gamma=float(model_config.get("focal_gamma", 2.0)),
        classification_prior=float(model_config.get("classification_prior", 0.01)),
        centerness_prior=float(model_config.get("centerness_prior", 0.01)),
        normalize_inputs=bool(model_config.get("normalize_inputs", True)),
        input_mean=tuple(model_config.get("input_mean", (0.485, 0.456, 0.406))),
        input_std=tuple(model_config.get("input_std", (0.229, 0.224, 0.225))),
    )
