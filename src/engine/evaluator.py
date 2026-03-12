"""Minimal evaluation engine for the baseline detector pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
import torch

from src.datasets.label_mapping import (
    remap_predictions_and_targets_for_cross_dataset,
    resolve_cross_dataset_label_mapping,
)
from src.datasets.base_dataset import normalize_class_name
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

    def _merge_tile_predictions(
        self,
        predictions: Sequence[Dict[str, Any]],
        merge_iou_threshold: float,
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
            if boxes.numel() > 0:
                boxes[:, 0] += offset_x
                boxes[:, 2] += offset_x
                boxes[:, 1] += offset_y
                boxes[:, 3] += offset_y

            grouped_predictions[source_image_id]["boxes"].append(boxes)
            grouped_predictions[source_image_id]["scores"].append(prediction["scores"].clone().cpu())
            grouped_predictions[source_image_id]["labels"].append(prediction["labels"].clone().cpu())

        merged_predictions: list[Dict[str, Any]] = []
        for source_image_id, grouped in grouped_predictions.items():
            boxes = torch.cat(grouped["boxes"], dim=0) if grouped["boxes"] else torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.cat(grouped["scores"], dim=0) if grouped["scores"] else torch.zeros((0,), dtype=torch.float32)
            labels = torch.cat(grouped["labels"], dim=0) if grouped["labels"] else torch.zeros((0,), dtype=torch.int64)

            kept_indices: list[int] = []
            for class_id in labels.unique(sorted=True).tolist():
                class_mask = labels == int(class_id)
                class_indices = class_mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
                if class_indices.size == 0:
                    continue
                class_keep_local = self._nms_numpy(
                    boxes[class_indices],
                    scores[class_indices],
                    iou_threshold=merge_iou_threshold,
                )
                kept_indices.extend(class_indices[index] for index in class_keep_local)

            if kept_indices:
                keep_tensor = torch.as_tensor(
                    kept_indices,
                    dtype=torch.long,
                )
                keep_tensor = keep_tensor[scores[keep_tensor].argsort(descending=True)]
                boxes = boxes[keep_tensor]
                scores = scores[keep_tensor]
                labels = labels[keep_tensor]
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
            summary["tile_merge_iou_threshold"] = tile_merge_iou_threshold
            summary["source_manifest_path"] = source_manifest_path
            summary["num_tile_images"] = len(predictions)
            summary["num_merged_images"] = len(metric_targets)
        if cross_dataset_summary is not None:
            summary.update(cross_dataset_summary)

        if save_outputs:
            summary_path = self.tables_dir / f"{experiment_name}_{resolved_split}_summary.json"
            per_class_path = self.tables_dir / f"{experiment_name}_{resolved_split}_per_class.csv"
            save_json(summary, summary_path)
            save_csv(metric_payload["per_class"], per_class_path)

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
