"""Hybrid CNN-Transformer detector with FCOS-like dense detection for ablations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.losses.detection_loss import (
    aligned_box_iou,
    generalized_box_iou_loss,
    pairwise_box_iou,
    sigmoid_focal_loss,
)
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
        atss_topk: int = 9,
        assignment_reference_scale: float = 4.0,
        box_loss_weight: float = 2.0,
        cls_loss_weight: float = 1.0,
        quality_loss_weight: float | None = None,
        centerness_loss_weight: float = 1.0,
        num_head_convs: int = 2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        classification_prior: float = 0.01,
        quality_prior: float | None = None,
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
        self.atss_topk = max(int(atss_topk), 1)
        self.assignment_reference_scale = float(assignment_reference_scale)
        self.box_loss_weight = float(box_loss_weight)
        self.cls_loss_weight = float(cls_loss_weight)
        resolved_quality_loss_weight = quality_loss_weight
        if resolved_quality_loss_weight is None:
            resolved_quality_loss_weight = centerness_loss_weight
        self.quality_loss_weight = float(resolved_quality_loss_weight)
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
            quality_prior=quality_prior if quality_prior is not None else centerness_prior,
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
        for level_name, feature in dense_outputs["quality_logits"].items():
            stride_y = float(image_height) / float(feature.shape[-2])
            stride_x = float(image_width) / float(feature.shape[-1])
            strides[level_name] = (stride_y, stride_x)
        return strides

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

    def _build_reference_boxes(
        self,
        centers: torch.Tensor,
        strides: tuple[float, float],
    ) -> torch.Tensor:
        reference_extent = self.assignment_reference_scale * max(strides)
        half_extent = reference_extent * 0.5
        return torch.stack(
            (
                centers[:, 0] - half_extent,
                centers[:, 1] - half_extent,
                centers[:, 0] + half_extent,
                centers[:, 1] + half_extent,
            ),
            dim=-1,
        )

    def _compute_point_weights(self, ltrb_targets: torch.Tensor) -> torch.Tensor:
        left_right = ltrb_targets[:, [0, 2]]
        top_bottom = ltrb_targets[:, [1, 3]]
        return torch.sqrt(
            (
                left_right.min(dim=-1).values
                / left_right.max(dim=-1).values.clamp(min=1e-6)
            )
            * (
                top_bottom.min(dim=-1).values
                / top_bottom.max(dim=-1).values.clamp(min=1e-6)
            )
        )

    def _split_flat_targets_by_level(
        self,
        level_names: Sequence[str],
        num_points_per_level: Mapping[str, int],
        labels: torch.Tensor,
        bbox_targets: torch.Tensor,
        target_boxes: torch.Tensor,
        point_weights: torch.Tensor,
        centers: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        split_targets: Dict[str, Dict[str, torch.Tensor]] = {}
        start_index = 0
        for level_name in level_names:
            end_index = start_index + num_points_per_level[level_name]
            split_targets[level_name] = {
                "labels": labels[start_index:end_index],
                "bbox_targets": bbox_targets[start_index:end_index],
                "target_boxes": target_boxes[start_index:end_index],
                "point_weights": point_weights[start_index:end_index],
                "centers": centers[start_index:end_index],
            }
            start_index = end_index
        return split_targets

    def _encode_targets_for_image(
        self,
        level_names: Sequence[str],
        level_metadata: Mapping[str, Dict[str, Any]],
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        flat_centers = []
        flat_reference_boxes = []
        num_points_per_level: Dict[str, int] = {}
        level_offsets: Dict[str, int] = {}
        start_index = 0

        for level_name in level_names:
            centers = level_metadata[level_name]["centers"]
            flat_centers.append(centers)
            flat_reference_boxes.append(level_metadata[level_name]["reference_boxes"])
            num_points_per_level[level_name] = int(centers.shape[0])
            level_offsets[level_name] = start_index
            start_index += int(centers.shape[0])

        all_centers = torch.cat(flat_centers, dim=0)
        all_reference_boxes = torch.cat(flat_reference_boxes, dim=0)
        total_points = all_centers.shape[0]
        device = all_centers.device

        labels_out = torch.full((total_points,), -1, dtype=torch.long, device=device)
        bbox_targets = torch.zeros((total_points, 4), dtype=torch.float32, device=device)
        target_boxes = torch.zeros((total_points, 4), dtype=torch.float32, device=device)
        point_weights = torch.zeros((total_points,), dtype=torch.float32, device=device)

        if boxes.numel() == 0:
            return self._split_flat_targets_by_level(
                level_names=level_names,
                num_points_per_level=num_points_per_level,
                labels=labels_out,
                bbox_targets=bbox_targets,
                target_boxes=target_boxes,
                point_weights=point_weights,
                centers=all_centers,
            )

        gt_centers = torch.stack(
            (
                (boxes[:, 0] + boxes[:, 2]) * 0.5,
                (boxes[:, 1] + boxes[:, 3]) * 0.5,
            ),
            dim=-1,
        )
        gt_areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
        assigned_gt = torch.full((total_points,), -1, dtype=torch.long, device=device)
        assigned_scores = torch.full((total_points,), -1.0, dtype=torch.float32, device=device)
        assigned_areas = torch.full((total_points,), float("inf"), dtype=torch.float32, device=device)

        for gt_index in range(boxes.shape[0]):
            gt_box = boxes[gt_index : gt_index + 1]
            gt_center = gt_centers[gt_index]
            candidate_indices = []

            for level_name in level_names:
                centers = level_metadata[level_name]["centers"]
                if centers.numel() == 0:
                    continue
                distances = torch.pow(centers[:, 0] - gt_center[0], 2) + torch.pow(centers[:, 1] - gt_center[1], 2)
                topk = min(self.atss_topk, int(centers.shape[0]))
                nearest = distances.topk(topk, largest=False).indices + level_offsets[level_name]
                candidate_indices.append(nearest)

            if not candidate_indices:
                continue

            candidate_indices = torch.cat(candidate_indices, dim=0)
            candidate_reference_boxes = all_reference_boxes[candidate_indices]
            candidate_ious = pairwise_box_iou(candidate_reference_boxes, gt_box).squeeze(1)
            threshold = candidate_ious.mean() + candidate_ious.std(unbiased=False)

            candidate_centers = all_centers[candidate_indices]
            inside_box = (
                (candidate_centers[:, 0] >= gt_box[0, 0])
                & (candidate_centers[:, 0] <= gt_box[0, 2])
                & (candidate_centers[:, 1] >= gt_box[0, 1])
                & (candidate_centers[:, 1] <= gt_box[0, 3])
            )
            positive_mask = (candidate_ious >= threshold) & inside_box
            positive_indices = candidate_indices[positive_mask]
            positive_scores = candidate_ious[positive_mask]

            if positive_indices.numel() == 0:
                inside_indices = candidate_indices[inside_box]
                inside_scores = candidate_ious[inside_box]
                if inside_indices.numel() > 0:
                    best_index = int(inside_scores.argmax().item())
                    positive_indices = inside_indices[best_index : best_index + 1]
                    positive_scores = inside_scores[best_index : best_index + 1]
                else:
                    best_index = int(candidate_ious.argmax().item())
                    positive_indices = candidate_indices[best_index : best_index + 1]
                    positive_scores = candidate_ious[best_index : best_index + 1]

            gt_area = gt_areas[gt_index]
            current_scores = assigned_scores[positive_indices]
            current_areas = assigned_areas[positive_indices]
            should_update = (positive_scores > current_scores) | (
                (positive_scores == current_scores) & (gt_area < current_areas)
            )
            update_indices = positive_indices[should_update]
            assigned_gt[update_indices] = gt_index
            assigned_scores[update_indices] = positive_scores[should_update]
            assigned_areas[update_indices] = gt_area

        positive_indices = (assigned_gt >= 0).nonzero(as_tuple=False).squeeze(1)
        if positive_indices.numel() > 0:
            matched_targets = assigned_gt[positive_indices]
            matched_boxes = boxes[matched_targets]
            matched_labels = labels[matched_targets]
            matched_centers = all_centers[positive_indices]
            matched_ltrb = torch.stack(
                (
                    matched_centers[:, 0] - matched_boxes[:, 0],
                    matched_centers[:, 1] - matched_boxes[:, 1],
                    matched_boxes[:, 2] - matched_centers[:, 0],
                    matched_boxes[:, 3] - matched_centers[:, 1],
                ),
                dim=-1,
            )
            labels_out[positive_indices] = matched_labels
            bbox_targets[positive_indices] = matched_ltrb
            target_boxes[positive_indices] = matched_boxes
            point_weights[positive_indices] = self._compute_point_weights(matched_ltrb)

        return self._split_flat_targets_by_level(
            level_names=level_names,
            num_points_per_level=num_points_per_level,
            labels=labels_out,
            bbox_targets=bbox_targets,
            target_boxes=target_boxes,
            point_weights=point_weights,
            centers=all_centers,
        )

    def _prepare_level_targets(
        self,
        dense_outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, torch.Tensor]],
        image_sizes: Sequence[tuple[int, int]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        level_names = list(dense_outputs["feature_levels"])
        target_bundle = {
            level_name: {
                "labels": [],
                "bbox_targets": [],
                "target_boxes": [],
                "point_weights": [],
                "centers": [],
            }
            for level_name in level_names
        }

        batch_size = len(image_sizes)
        for image_index in range(batch_size):
            image_height, image_width = image_sizes[image_index]
            level_metadata: Dict[str, Dict[str, Any]] = {}

            for level_name in level_names:
                cls_logits = dense_outputs["cls_logits"][level_name]
                _, _, height, width = cls_logits.shape
                stride_y = float(image_height) / float(height)
                stride_x = float(image_width) / float(width)
                centers = self._compute_level_centers(
                    height=height,
                    width=width,
                    stride_y=stride_y,
                    stride_x=stride_x,
                    device=cls_logits.device,
                )
                level_metadata[level_name] = {
                    "centers": centers,
                    "reference_boxes": self._build_reference_boxes(centers, (stride_y, stride_x)),
                }

            image_targets = self._encode_targets_for_image(
                level_names=level_names,
                level_metadata=level_metadata,
                boxes=targets[image_index]["boxes"],
                labels=targets[image_index]["labels"] - 1,
            )

            for level_name in level_names:
                for key in target_bundle[level_name]:
                    target_bundle[level_name][key].append(image_targets[level_name][key])

        for level_name in level_names:
            for key in target_bundle[level_name]:
                target_bundle[level_name][key] = torch.stack(target_bundle[level_name][key], dim=0)
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
        quality_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        total_positive = 0
        box_weight_sum = torch.tensor(0.0, device=next(self.parameters()).device)

        for level_name in dense_outputs["feature_levels"]:
            cls_logits = dense_outputs["cls_logits"][level_name].permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            bbox_regression = dense_outputs["bbox_regression"][level_name].permute(0, 2, 3, 1).reshape(-1, 4)
            quality_logits = dense_outputs["quality_logits"][level_name].permute(0, 2, 3, 1).reshape(-1)

            level_targets = target_bundle[level_name]
            labels = level_targets["labels"].reshape(-1)
            target_boxes = level_targets["target_boxes"].reshape(-1, 4)
            centers = level_targets["centers"].reshape(-1, 2)
            point_weights = level_targets["point_weights"].reshape(-1)
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
                box_weights = point_weights[positive_mask]
                box_losses = generalized_box_iou_loss(
                    pred_boxes,
                    target_boxes[positive_mask],
                    reduction="none",
                )
                box_loss = box_loss + (box_losses * box_weights).sum()
                box_weight_sum = box_weight_sum + box_weights.sum()
                quality_targets = aligned_box_iou(pred_boxes.detach(), target_boxes[positive_mask]).clamp(0.0, 1.0)
                quality_loss = quality_loss + F.binary_cross_entropy_with_logits(
                    quality_logits[positive_mask],
                    quality_targets,
                    reduction="sum",
                )

        normalizer = max(total_positive, 1)
        box_normalizer = torch.clamp(box_weight_sum, min=1.0)
        return {
            "loss_cls": (cls_loss / normalizer) * self.cls_loss_weight,
            "loss_box_reg": (box_loss / box_normalizer) * self.box_loss_weight,
            "loss_quality": (quality_loss / normalizer) * self.quality_loss_weight,
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
                quality_logits = dense_outputs["quality_logits"][level_name][batch_index, 0]
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
                quality_scores = torch.sigmoid(quality_logits).reshape(-1, 1)
                combined_scores = cls_scores * quality_scores
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
        atss_topk=int(model_config.get("atss_topk", 9)),
        assignment_reference_scale=float(model_config.get("assignment_reference_scale", 4.0)),
        box_loss_weight=float(model_config.get("box_loss_weight", 2.0)),
        cls_loss_weight=float(model_config.get("cls_loss_weight", 1.0)),
        quality_loss_weight=(
            float(model_config["quality_loss_weight"])
            if "quality_loss_weight" in model_config
            else None
        ),
        centerness_loss_weight=float(model_config.get("centerness_loss_weight", 1.0)),
        num_head_convs=int(model_config.get("num_head_convs", 2)),
        focal_alpha=float(model_config.get("focal_alpha", 0.25)),
        focal_gamma=float(model_config.get("focal_gamma", 2.0)),
        classification_prior=float(model_config.get("classification_prior", 0.01)),
        quality_prior=(
            float(model_config["quality_prior"])
            if "quality_prior" in model_config
            else None
        ),
        centerness_prior=float(model_config.get("centerness_prior", 0.01)),
        normalize_inputs=bool(model_config.get("normalize_inputs", True)),
        input_mean=tuple(model_config.get("input_mean", (0.485, 0.456, 0.406))),
        input_std=tuple(model_config.get("input_std", (0.229, 0.224, 0.225))),
    )
