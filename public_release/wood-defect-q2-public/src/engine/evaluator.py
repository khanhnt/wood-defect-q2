"""Minimal evaluation engine for the baseline detector pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch

from src.datasets.label_mapping import (
    remap_predictions_and_targets_for_cross_dataset,
    resolve_cross_dataset_label_mapping,
)
from src.datasets.base_dataset import normalize_class_name, resolve_small_defect_rule
from src.datasets.manifest_detection_dataset import build_detection_dataloader, load_manifest_records
from src.metrics.detection_metrics import _to_numpy, box_iou_numpy, compute_detection_metrics
from src.utils.io import ensure_dir, save_csv, save_json, save_jsonl
from src.utils.logger import setup_logger

logger = setup_logger()


class Evaluator:
    """Evaluate a detection model on a manifest-backed dataset."""

    def __init__(self, model: Any, config: dict) -> None:
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.get("device", "cpu"))
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.tables_dir = ensure_dir(self.output_dir / "tables")

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_eval_loader(self) -> tuple[Any, Dict[str, Any]]:
        dataset_cfg = self.config.get("dataset", {})
        eval_cfg = self.config.get("evaluation", {})
        split_cfg = self.config.get("dataset_split", {})

        dataset_source = (
            dataset_cfg.get("eval")
            or dataset_cfg.get("val")
            or dataset_cfg.get("train")
        )
        if dataset_source is None:
            raise ValueError("Evaluation config requires dataset.eval, dataset.val, or dataset.train.")

        split_name = dataset_cfg.get("split", "val")
        batch_size = int(eval_cfg.get("batch_size", 1))
        num_workers = int(eval_cfg.get("num_workers", 0))
        max_samples = eval_cfg.get("max_samples")

        return build_detection_dataloader(
            dataset_config_or_path=dataset_source,
            split=split_name,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            split_seed=int(split_cfg.get("seed", self.config.get("seed", 42))),
            train_ratio=float(split_cfg.get("train_ratio", 0.8)),
            val_ratio=float(split_cfg.get("val_ratio", 0.2)),
            max_samples=int(max_samples) if max_samples is not None else None,
        )

    def _load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        return checkpoint

    def _infer_source_manifest_path(self, dataset_config: Dict[str, Any]) -> Path | None:
        explicit_path = self.config.get("evaluation", {}).get("source_manifest_path")
        if explicit_path:
            candidate = Path(explicit_path)
            if candidate.exists():
                return candidate

        dataset_name = dataset_config.get("dataset_name")
        if not dataset_name:
            return None
        candidate = Path("data/processed") / f"{dataset_name}_manifest.jsonl"
        if candidate.exists():
            return candidate
        return None

    def _nms_numpy(self, boxes: Any, scores: Any, iou_threshold: float) -> list[int]:
        boxes_array = _to_numpy(boxes).astype("float32")
        scores_array = _to_numpy(scores).astype("float32")
        if boxes_array.size == 0:
            return []

        order = scores_array.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            ious = box_iou_numpy(boxes_array[current : current + 1], boxes_array[remaining])[0]
            order = remaining[ious <= float(iou_threshold)]

        return keep

    def _edge_score_weights(
        self,
        boxes: torch.Tensor,
        tile_width: int,
        tile_height: int,
        margin_px: float,
        edge_penalty: float,
    ) -> torch.Tensor:
        if boxes.numel() == 0 or margin_px <= 0:
            return torch.ones((len(boxes),), dtype=torch.float32)

        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        tile_width_f = float(tile_width)
        tile_height_f = float(tile_height)
        distances = torch.stack(
            [
                centers_x,
                centers_y,
                tile_width_f - centers_x,
                tile_height_f - centers_y,
            ],
            dim=1,
        ).min(dim=1).values
        normalized = torch.clamp(distances / float(margin_px), min=0.0, max=1.0)
        base_penalty = float(edge_penalty)
        return base_penalty + (1.0 - base_penalty) * normalized

    def _weighted_box_fusion_numpy(
        self,
        boxes: Any,
        scores: Any,
        iou_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes_array = _to_numpy(boxes).astype("float32")
        scores_array = _to_numpy(scores).astype("float32")
        if boxes_array.size == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        order = scores_array.argsort()[::-1]
        fused_boxes: list[np.ndarray] = []
        fused_scores: list[float] = []

        while order.size > 0:
            anchor = int(order[0])
            if order.size == 1:
                cluster_indices = np.asarray([anchor], dtype=np.int64)
                order = np.zeros((0,), dtype=np.int64)
            else:
                remaining = order[1:]
                ious = box_iou_numpy(boxes_array[anchor : anchor + 1], boxes_array[remaining])[0]
                matched = remaining[ious >= float(iou_threshold)]
                cluster_indices = np.concatenate(
                    [np.asarray([anchor], dtype=np.int64), matched.astype(np.int64)]
                )
                order = remaining[ious < float(iou_threshold)]

            cluster_boxes = boxes_array[cluster_indices]
            cluster_scores = np.clip(scores_array[cluster_indices], a_min=1e-6, a_max=None)
            weighted_box = (cluster_boxes * cluster_scores[:, None]).sum(axis=0) / cluster_scores.sum()
            fused_boxes.append(weighted_box.astype(np.float32))
            fused_scores.append(float(cluster_scores.max()))

        return np.stack(fused_boxes, axis=0), np.asarray(fused_scores, dtype=np.float32)

    def _merge_tile_predictions(
        self,
        predictions: Sequence[Dict[str, Any]],
        merge_iou_threshold: float,
        merge_method: str,
        pre_merge_score_threshold: float,
        post_merge_score_threshold: float,
        edge_margin_px: float,
        edge_penalty: float,
        max_detections: int,
    ) -> list[Dict[str, Any]]:
        grouped_predictions: Dict[str, Dict[str, list[Any]]] = {}

        for prediction in predictions:
            source_image_id = prediction.get("source_image_id") or prediction["image_id"]
            tile_origin = prediction.get("tile_origin_xy") or [0, 0]
            offset_x = float(tile_origin[0])
            offset_y = float(tile_origin[1])

            grouped_predictions.setdefault(
                source_image_id,
                {"boxes": [], "scores": [], "labels": []},
            )

            boxes = prediction["boxes"].clone().cpu()
            scores = prediction["scores"].clone().cpu()
            labels = prediction["labels"].clone().cpu()
            tile_width = int(prediction.get("tile_width", 0) or 0)
            tile_height = int(prediction.get("tile_height", 0) or 0)

            if boxes.numel() > 0 and tile_width > 0 and tile_height > 0:
                edge_weights = self._edge_score_weights(
                    boxes=boxes,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    margin_px=edge_margin_px,
                    edge_penalty=edge_penalty,
                )
                scores = scores * edge_weights

            keep_mask = scores >= float(pre_merge_score_threshold)
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]

            if boxes.numel() > 0:
                boxes[:, 0] += offset_x
                boxes[:, 2] += offset_x
                boxes[:, 1] += offset_y
                boxes[:, 3] += offset_y

            grouped_predictions[source_image_id]["boxes"].append(boxes)
            grouped_predictions[source_image_id]["scores"].append(scores)
            grouped_predictions[source_image_id]["labels"].append(labels)

        merged_predictions: list[Dict[str, Any]] = []
        for source_image_id, grouped in grouped_predictions.items():
            boxes = torch.cat(grouped["boxes"], dim=0) if grouped["boxes"] else torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.cat(grouped["scores"], dim=0) if grouped["scores"] else torch.zeros((0,), dtype=torch.float32)
            labels = torch.cat(grouped["labels"], dim=0) if grouped["labels"] else torch.zeros((0,), dtype=torch.int64)

            merged_boxes_per_class: list[torch.Tensor] = []
            merged_scores_per_class: list[torch.Tensor] = []
            merged_labels_per_class: list[torch.Tensor] = []

            for class_id in labels.unique(sorted=True).tolist():
                class_mask = labels == int(class_id)
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]
                if class_boxes.numel() == 0:
                    continue

                if merge_method == "wbf":
                    fused_boxes, fused_scores = self._weighted_box_fusion_numpy(
                        boxes=class_boxes,
                        scores=class_scores,
                        iou_threshold=merge_iou_threshold,
                    )
                    class_boxes = torch.as_tensor(fused_boxes, dtype=torch.float32)
                    class_scores = torch.as_tensor(fused_scores, dtype=torch.float32)
                else:
                    class_keep = self._nms_numpy(
                        class_boxes,
                        class_scores,
                        iou_threshold=merge_iou_threshold,
                    )
                    keep_tensor = torch.as_tensor(class_keep, dtype=torch.long)
                    class_boxes = class_boxes[keep_tensor]
                    class_scores = class_scores[keep_tensor]

                score_mask = class_scores >= float(post_merge_score_threshold)
                class_boxes = class_boxes[score_mask]
                class_scores = class_scores[score_mask]
                if class_boxes.numel() == 0:
                    continue

                class_labels = torch.full(
                    (class_boxes.shape[0],),
                    int(class_id),
                    dtype=torch.int64,
                )
                merged_boxes_per_class.append(class_boxes)
                merged_scores_per_class.append(class_scores)
                merged_labels_per_class.append(class_labels)

            if merged_boxes_per_class:
                boxes = torch.cat(merged_boxes_per_class, dim=0)
                scores = torch.cat(merged_scores_per_class, dim=0)
                labels = torch.cat(merged_labels_per_class, dim=0)
                order = scores.argsort(descending=True)
                if max_detections > 0:
                    order = order[: int(max_detections)]
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                scores = torch.zeros((0,), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            merged_predictions.append(
                {
                    "image_id": source_image_id,
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                }
            )

        return merged_predictions

    def _load_source_level_targets(
        self,
        dataset_config: Dict[str, Any],
        source_image_ids: Sequence[str],
        target_class_names: Sequence[str],
    ) -> tuple[list[Dict[str, Any]], list[str], str]:
        source_manifest_path = self._infer_source_manifest_path(dataset_config)
        if source_manifest_path is None:
            raise FileNotFoundError(
                "Tile-aware merge evaluation needs a source-level manifest. "
                "Set evaluation.source_manifest_path or place the source manifest under data/processed."
            )

        source_records, source_meta = load_manifest_records(
            dataset_config_or_path={"manifest_path": str(source_manifest_path), "dataset_name": dataset_config.get("dataset_name")},
            split=None,
        )
        source_ids = set(source_image_ids)
        target_class_to_id = {
            normalize_class_name(class_name): class_id
            for class_id, class_name in enumerate(target_class_names)
        }
        filtered_targets: list[Dict[str, Any]] = []
        skipped_annotations = 0

        for record in source_records:
            if record["image_id"] not in source_ids:
                continue

            boxes = []
            labels = []
            for annotation in record.get("annotations", []):
                normalized_class_name = normalize_class_name(annotation["class_name"])
                if normalized_class_name not in target_class_to_id:
                    skipped_annotations += 1
                    continue
                x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
                width = float(record["width"])
                height = float(record["height"])
                boxes.append(
                    [
                        float(x1) * width,
                        float(y1) * height,
                        float(x2) * width,
                        float(y2) * height,
                    ]
                )
                labels.append(int(target_class_to_id[normalized_class_name]))

            filtered_targets.append(
                {
                    "image_id": record["image_id"],
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.as_tensor(labels, dtype=torch.int64),
                }
            )

        if not filtered_targets:
            raise ValueError(
                "Tile-merge evaluation could not match any source-level targets. "
                "Check source_manifest_path and source_image_id fields in the processed manifest."
            )

        if skipped_annotations > 0:
            logger.warning(
                "Tile-merge evaluation skipped %d source annotations due to missing class-name mapping.",
                skipped_annotations,
            )

        return filtered_targets, list(target_class_names), str(source_manifest_path)

    def _prediction_to_serializable(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "image_id": prediction["image_id"],
            "boxes": torch.as_tensor(prediction["boxes"]).cpu().tolist(),
            "labels": torch.as_tensor(prediction["labels"]).cpu().tolist(),
            "scores": torch.as_tensor(prediction["scores"]).cpu().tolist(),
        }
        if prediction.get("source_image_id") is not None:
            payload["source_image_id"] = prediction["source_image_id"]
        if prediction.get("tile_origin_xy") is not None:
            payload["tile_origin_xy"] = prediction["tile_origin_xy"]
        return payload

    def _annotation_is_small(
        self,
        annotation: Dict[str, Any],
        record_width: int,
        record_height: int,
        small_defect_rule: Dict[str, Any],
    ) -> bool:
        if bool(annotation.get("is_small_defect", False)):
            return True

        checks: list[bool] = []
        if small_defect_rule.get("min_area_ratio") is not None:
            checks.append(float(annotation.get("bbox_area_norm", 0.0)) <= float(small_defect_rule["min_area_ratio"]))
        if small_defect_rule.get("min_width_px") is not None and record_width > 0:
            width_px = float(annotation.get("bbox_width_norm", 0.0)) * float(record_width)
            checks.append(width_px <= float(small_defect_rule["min_width_px"]))
        if small_defect_rule.get("min_height_px") is not None and record_height > 0:
            height_px = float(annotation.get("bbox_height_norm", 0.0)) * float(record_height)
            checks.append(height_px <= float(small_defect_rule["min_height_px"]))

        if not small_defect_rule["enabled"] or not checks:
            return False
        if small_defect_rule["combine"] == "all":
            return all(checks)
        return any(checks)

    def _build_small_defect_eval_payloads(
        self,
        dataset_config: Dict[str, Any],
        predictions: Sequence[Dict[str, Any]],
        target_class_names: Sequence[str],
    ) -> Dict[str, Any] | None:
        small_defect_cfg = dataset_config.get("small_defect")
        if not small_defect_cfg:
            return None

        small_defect_rule = resolve_small_defect_rule(small_defect_cfg)
        manifest_records, _ = load_manifest_records(
            dataset_config_or_path=dataset_config,
            split=None,
        )
        prediction_by_image = {prediction["image_id"]: prediction for prediction in predictions}
        target_class_to_id = {
            normalize_class_name(class_name): class_id
            for class_id, class_name in enumerate(target_class_names)
        }

        small_target_targets: list[Dict[str, Any]] = []
        small_target_predictions: list[Dict[str, Any]] = []
        small_image_targets: list[Dict[str, Any]] = []
        small_image_predictions: list[Dict[str, Any]] = []

        for record in manifest_records:
            image_id = record["image_id"]
            prediction = prediction_by_image.get(image_id)
            if prediction is None:
                continue

            record_width = int(record.get("width", 0) or 0)
            record_height = int(record.get("height", 0) or 0)
            all_boxes = []
            all_labels = []
            small_boxes = []
            small_labels = []

            for annotation in record.get("annotations", []):
                normalized_class_name = normalize_class_name(annotation["class_name"])
                if normalized_class_name not in target_class_to_id:
                    continue
                x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
                box = [
                    float(x1) * record_width,
                    float(y1) * record_height,
                    float(x2) * record_width,
                    float(y2) * record_height,
                ]
                class_id = int(target_class_to_id[normalized_class_name])
                all_boxes.append(box)
                all_labels.append(class_id)

                if self._annotation_is_small(
                    annotation=annotation,
                    record_width=record_width,
                    record_height=record_height,
                    small_defect_rule=small_defect_rule,
                ):
                    small_boxes.append(box)
                    small_labels.append(class_id)

            if not all_boxes:
                continue

            if small_boxes:
                small_target_targets.append(
                    {
                        "image_id": image_id,
                        "boxes": torch.as_tensor(small_boxes, dtype=torch.float32).reshape(-1, 4),
                        "labels": torch.as_tensor(small_labels, dtype=torch.int64),
                    }
                )
                small_target_predictions.append(prediction)

                small_image_targets.append(
                    {
                        "image_id": image_id,
                        "boxes": torch.as_tensor(all_boxes, dtype=torch.float32).reshape(-1, 4),
                        "labels": torch.as_tensor(all_labels, dtype=torch.int64),
                    }
                )
                small_image_predictions.append(prediction)

        if not small_target_targets:
            return None

        small_target_metrics = compute_detection_metrics(
            predictions=small_target_predictions,
            targets=small_target_targets,
            class_names=target_class_names,
            score_threshold=float(self.config.get("evaluation", {}).get("score_threshold", 0.05)),
        )
        small_image_metrics = compute_detection_metrics(
            predictions=small_image_predictions,
            targets=small_image_targets,
            class_names=target_class_names,
            score_threshold=float(self.config.get("evaluation", {}).get("score_threshold", 0.05)),
        )
        return {
            "rule": small_defect_rule,
            "small_target": small_target_metrics,
            "small_image": small_image_metrics,
        }

    def _load_summary_json(self, path: str | Path | None) -> Dict[str, Any] | None:
        if not path:
            return None
        summary_path = Path(path)
        if not summary_path.exists():
            return None
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def _resolve_in_domain_summary(self, checkpoint: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        eval_cfg = self.config.get("evaluation", {})
        explicit_path = eval_cfg.get("in_domain_summary_path")
        if explicit_path:
            summary = self._load_summary_json(explicit_path)
            if summary is not None:
                return summary

        train_experiment_name = None
        if checkpoint is not None:
            train_experiment_name = checkpoint.get("config", {}).get("experiment_name")
        train_experiment_name = train_experiment_name or self.config.get("train_experiment_name")

        candidate_paths = []
        if train_experiment_name:
            candidate_paths.append(self.tables_dir / f"{train_experiment_name}_best_val_summary.json")
        candidate_paths.append(self.tables_dir / "baseline_detector_main_eval_val_summary.json")

        for candidate_path in candidate_paths:
            summary = self._load_summary_json(candidate_path)
            if summary is not None:
                return summary
        return None

    def _build_cross_dataset_summary(
        self,
        mapping_report: Dict[str, Any],
        remap_counts: Dict[str, int],
        source_class_names: Sequence[str],
        target_class_names: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "source_class_names": list(source_class_names),
            "target_class_names": list(target_class_names),
            "evaluated_class_names": list(mapping_report["mapped_class_names"]),
            "mapped_target_classes": list(mapping_report["mapped_target_classes"]),
            "ignored_target_classes": list(mapping_report["ignored_target_classes"]),
            "unmatched_target_classes": list(mapping_report["unmatched_target_classes"]),
            "unmatched_source_classes": list(mapping_report["unmatched_source_classes"]),
            "ignored_prediction_count": int(remap_counts["ignored_prediction_count"]),
            "ignored_target_annotation_count": int(remap_counts["ignored_target_annotation_count"]),
            "mapped_prediction_count": int(remap_counts["mapped_prediction_count"]),
            "mapped_target_annotation_count": int(remap_counts["mapped_target_annotation_count"]),
            "assumption": "Only mapped overlapping classes are evaluated. Unmapped source predictions are ignored for cross-dataset scoring.",
        }

    def _export_cross_dataset_reports(
        self,
        experiment_name: str,
        split_name: str,
        current_summary: Dict[str, Any],
        in_domain_summary: Dict[str, Any] | None,
        mapping_report: Dict[str, Any],
    ) -> None:
        mapping_df = mapping_report["mapping_table"]
        mapping_path = self.tables_dir / f"{experiment_name}_{split_name}_label_mapping.csv"
        save_csv(mapping_df, mapping_path)

        comparison_rows = []
        if in_domain_summary is not None:
            comparison_rows.append(
                {
                    "evaluation_scope": "in_domain",
                    "dataset_label": self.config.get("evaluation", {}).get("in_domain_dataset_label", "main_validation"),
                    "split": in_domain_summary.get("split", "val"),
                    "mAP50": in_domain_summary.get("mAP50"),
                    "mAP50_95": in_domain_summary.get("mAP50_95"),
                    "precision50": in_domain_summary.get("precision50"),
                    "recall50": in_domain_summary.get("recall50"),
                    "num_images": in_domain_summary.get("num_images"),
                    "num_targets": in_domain_summary.get("num_targets"),
                    "evaluated_classes": ";".join(in_domain_summary.get("class_names", [])),
                    "mapped_classes": "",
                    "ignored_classes": "",
                    "unmatched_classes": "",
                }
            )

        cross_unmatched = sorted(
            set(current_summary.get("unmatched_target_classes", []))
            | set(current_summary.get("unmatched_source_classes", []))
        )
        comparison_rows.append(
            {
                "evaluation_scope": "cross_dataset",
                "dataset_label": current_summary.get("dataset_name", "cross_dataset"),
                "split": split_name,
                "mAP50": current_summary.get("mAP50"),
                "mAP50_95": current_summary.get("mAP50_95"),
                "precision50": current_summary.get("precision50"),
                "recall50": current_summary.get("recall50"),
                "num_images": current_summary.get("num_images"),
                "num_targets": current_summary.get("num_targets"),
                "evaluated_classes": ";".join(current_summary.get("evaluated_class_names", [])),
                "mapped_classes": ";".join(current_summary.get("mapped_target_classes", [])),
                "ignored_classes": ";".join(current_summary.get("ignored_target_classes", [])),
                "unmatched_classes": ";".join(cross_unmatched),
            }
        )
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_path = self.tables_dir / f"{experiment_name}_{split_name}_comparison.csv"
        save_csv(comparison_df, comparison_path)

    def evaluate(
        self,
        data_loader: Any | None = None,
        data_meta: Dict[str, Any] | None = None,
        split_name: str | None = None,
        checkpoint_path: str | Path | None = None,
        experiment_name: str | None = None,
        save_outputs: bool = True,
    ) -> Dict[str, Any]:
        """Run model evaluation and optionally export compact summary tables."""
        checkpoint = None
        if checkpoint_path is not None:
            checkpoint = self._load_checkpoint(checkpoint_path)

        if data_loader is None or data_meta is None:
            data_loader, data_meta = self._build_eval_loader()

        resolved_split = split_name or data_meta.get("split", "eval")
        experiment_name = experiment_name or self.config.get("experiment_name", "baseline_detector")
        score_threshold = float(self.config.get("evaluation", {}).get("score_threshold", 0.05))
        eval_cfg = self.config.get("evaluation", {})
        tile_merge_enabled = bool(eval_cfg.get("tile_merge", False))
        tile_merge_iou_threshold = float(eval_cfg.get("tile_merge_iou_threshold", 0.5))
        tile_merge_method = str(eval_cfg.get("tile_merge_method", "wbf")).lower()
        if tile_merge_method not in {"nms", "wbf"}:
            raise ValueError("evaluation.tile_merge_method must be either 'nms' or 'wbf'.")
        tile_pre_merge_score_threshold = float(eval_cfg.get("tile_pre_merge_score_threshold", score_threshold))
        tile_post_merge_score_threshold = float(eval_cfg.get("tile_post_merge_score_threshold", score_threshold))
        tile_edge_margin_px = float(eval_cfg.get("tile_edge_margin_px", 32.0))
        tile_edge_penalty = float(eval_cfg.get("tile_edge_penalty", 0.6))
        tile_max_detections = int(eval_cfg.get("tile_max_detections", self.config.get("model", {}).get("max_detections", 100)))

        self.model.to(self.device)
        self.model.eval()

        predictions: list[Dict[str, Any]] = []
        targets: list[Dict[str, Any]] = []

        with torch.no_grad():
            for images, batch_targets, metadata in data_loader:
                images = [image.to(self.device) for image in images]
                outputs = self.model(images)

                for output, target, meta in zip(outputs, batch_targets, metadata):
                    predictions.append(
                        {
                            "image_id": meta["image_id"],
                            "boxes": output["boxes"].detach().cpu(),
                            "labels": torch.clamp(output["labels"].detach().cpu() - 1, min=0),
                            "scores": output["scores"].detach().cpu(),
                            "source_image_id": meta.get("source_image_id"),
                            "tile_origin_xy": meta.get("tile_origin_xy"),
                            "tile_width": meta.get("record_width"),
                            "tile_height": meta.get("record_height"),
                        }
                    )
                    targets.append(
                        {
                            "image_id": meta["image_id"],
                            "boxes": target["boxes"].detach().cpu(),
                            "labels": torch.clamp(target["labels"].detach().cpu() - 1, min=0),
                        }
                    )

        dataset_config = data_meta.get("dataset_config", {})
        target_class_names = list(data_meta["class_names"])
        source_class_names = list((checkpoint or {}).get("class_names") or target_class_names)
        metric_predictions = predictions
        metric_targets = targets
        metric_class_names = target_class_names
        mapping_report = None
        cross_dataset_summary = None
        source_manifest_path = None
        small_defect_eval_payload = None

        if tile_merge_enabled:
            if bool(eval_cfg.get("compute_cross_dataset", False)):
                raise ValueError("Tile-merge evaluation is not supported together with cross-dataset remapping.")

            source_image_ids = [
                prediction.get("source_image_id") or prediction["image_id"]
                for prediction in predictions
            ]
            metric_predictions = self._merge_tile_predictions(
                predictions=predictions,
                merge_iou_threshold=tile_merge_iou_threshold,
                merge_method=tile_merge_method,
                pre_merge_score_threshold=tile_pre_merge_score_threshold,
                post_merge_score_threshold=tile_post_merge_score_threshold,
                edge_margin_px=tile_edge_margin_px,
                edge_penalty=tile_edge_penalty,
                max_detections=tile_max_detections,
            )
            metric_targets, metric_class_names, source_manifest_path = self._load_source_level_targets(
                dataset_config=dataset_config,
                source_image_ids=source_image_ids,
                target_class_names=target_class_names,
            )

        if bool(eval_cfg.get("compute_cross_dataset", False)):
            mapping_report = resolve_cross_dataset_label_mapping(
                source_class_names=source_class_names,
                target_class_names=target_class_names,
                label_mapping=dataset_config.get("label_mapping"),
            )
            metric_predictions, metric_targets, remap_counts = remap_predictions_and_targets_for_cross_dataset(
                predictions=predictions,
                targets=targets,
                mapping_report=mapping_report,
            )
            metric_class_names = list(mapping_report["mapped_class_names"])
            if not metric_class_names:
                raise ValueError("Cross-dataset evaluation produced no mapped classes to score.")
            cross_dataset_summary = self._build_cross_dataset_summary(
                mapping_report=mapping_report,
                remap_counts=remap_counts,
                source_class_names=source_class_names,
                target_class_names=target_class_names,
            )
        elif bool(eval_cfg.get("compute_small_defect_eval", False)) and not tile_merge_enabled:
            small_defect_eval_payload = self._build_small_defect_eval_payloads(
                dataset_config=dataset_config,
                predictions=predictions,
                target_class_names=target_class_names,
            )

        metric_payload = compute_detection_metrics(
            predictions=metric_predictions,
            targets=metric_targets,
            class_names=metric_class_names,
            score_threshold=score_threshold,
        )

        summary = dict(metric_payload["summary"])
        summary.update(
            {
                "experiment_name": experiment_name,
                "dataset_name": dataset_config.get("dataset_name", "unknown_dataset"),
                "split": resolved_split,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "class_names": list(metric_class_names),
                "evaluation_mode": "cross_dataset" if mapping_report is not None else "in_domain",
                "tile_merge": tile_merge_enabled,
            }
        )
        if tile_merge_enabled:
            summary["evaluation_mode"] = "tile_merge_in_domain"
            summary["tile_merge_method"] = tile_merge_method
            summary["tile_merge_iou_threshold"] = tile_merge_iou_threshold
            summary["tile_pre_merge_score_threshold"] = tile_pre_merge_score_threshold
            summary["tile_post_merge_score_threshold"] = tile_post_merge_score_threshold
            summary["tile_edge_margin_px"] = tile_edge_margin_px
            summary["tile_edge_penalty"] = tile_edge_penalty
            summary["tile_max_detections"] = tile_max_detections
            summary["source_manifest_path"] = source_manifest_path
            summary["num_tile_images"] = len(predictions)
            summary["num_merged_images"] = len(metric_targets)
        if cross_dataset_summary is not None:
            summary.update(cross_dataset_summary)
        if small_defect_eval_payload is not None:
            summary["small_defect_rule"] = small_defect_eval_payload["rule"]
            summary["small_target_mAP50"] = small_defect_eval_payload["small_target"]["summary"]["mAP50"]
            summary["small_target_mAP50_95"] = small_defect_eval_payload["small_target"]["summary"]["mAP50_95"]
            summary["small_target_num_images"] = small_defect_eval_payload["small_target"]["summary"]["num_images"]
            summary["small_target_num_targets"] = small_defect_eval_payload["small_target"]["summary"]["num_targets"]
            summary["small_image_subset_mAP50"] = small_defect_eval_payload["small_image"]["summary"]["mAP50"]
            summary["small_image_subset_mAP50_95"] = small_defect_eval_payload["small_image"]["summary"]["mAP50_95"]
            summary["small_image_subset_num_images"] = small_defect_eval_payload["small_image"]["summary"]["num_images"]
            summary["small_image_subset_num_targets"] = small_defect_eval_payload["small_image"]["summary"]["num_targets"]

        if save_outputs:
            summary_path = self.tables_dir / f"{experiment_name}_{resolved_split}_summary.json"
            per_class_path = self.tables_dir / f"{experiment_name}_{resolved_split}_per_class.csv"
            save_json(summary, summary_path)
            save_csv(metric_payload["per_class"], per_class_path)

            if small_defect_eval_payload is not None:
                small_target_summary_path = self.tables_dir / f"{experiment_name}_{resolved_split}_small_target_summary.json"
                small_target_per_class_path = self.tables_dir / f"{experiment_name}_{resolved_split}_small_target_per_class.csv"
                small_image_summary_path = self.tables_dir / f"{experiment_name}_{resolved_split}_small_image_subset_summary.json"
                small_image_per_class_path = self.tables_dir / f"{experiment_name}_{resolved_split}_small_image_subset_per_class.csv"
                save_json(small_defect_eval_payload["small_target"]["summary"], small_target_summary_path)
                save_csv(small_defect_eval_payload["small_target"]["per_class"], small_target_per_class_path)
                save_json(small_defect_eval_payload["small_image"]["summary"], small_image_summary_path)
                save_csv(small_defect_eval_payload["small_image"]["per_class"], small_image_per_class_path)

            if self.config.get("evaluation", {}).get("save_predictions", False):
                predictions_path = self.tables_dir / f"{experiment_name}_{resolved_split}_predictions.jsonl"
                serializable_predictions = metric_predictions if tile_merge_enabled else predictions
                save_jsonl(
                    [self._prediction_to_serializable(prediction) for prediction in serializable_predictions],
                    predictions_path,
                )

            if mapping_report is not None:
                in_domain_summary = self._resolve_in_domain_summary(checkpoint=checkpoint)
                self._export_cross_dataset_reports(
                    experiment_name=experiment_name,
                    split_name=resolved_split,
                    current_summary=summary,
                    in_domain_summary=in_domain_summary,
                    mapping_report=mapping_report,
                )

            logger.info("Saved evaluation summary to %s", summary_path)

        return {
            "summary": summary,
            "per_class": metric_payload["per_class"],
        }
