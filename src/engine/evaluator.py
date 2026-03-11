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
from src.datasets.manifest_detection_dataset import build_detection_dataloader
from src.metrics.detection_metrics import compute_detection_metrics
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

    def _prediction_to_serializable(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image_id": prediction["image_id"],
            "boxes": torch.as_tensor(prediction["boxes"]).cpu().tolist(),
            "labels": torch.as_tensor(prediction["labels"]).cpu().tolist(),
            "scores": torch.as_tensor(prediction["scores"]).cpu().tolist(),
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
            }
        )
        if cross_dataset_summary is not None:
            summary.update(cross_dataset_summary)

        if save_outputs:
            summary_path = self.tables_dir / f"{experiment_name}_{resolved_split}_summary.json"
            per_class_path = self.tables_dir / f"{experiment_name}_{resolved_split}_per_class.csv"
            save_json(summary, summary_path)
            save_csv(metric_payload["per_class"], per_class_path)

            if self.config.get("evaluation", {}).get("save_predictions", False):
                predictions_path = self.tables_dir / f"{experiment_name}_{resolved_split}_predictions.jsonl"
                save_jsonl(
                    [self._prediction_to_serializable(prediction) for prediction in predictions],
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
