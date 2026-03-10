"""Minimal evaluation engine for the baseline detector pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import torch

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
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if data_loader is None or data_meta is None:
            data_loader, data_meta = self._build_eval_loader()

        resolved_split = split_name or data_meta.get("split", "eval")
        experiment_name = experiment_name or self.config.get("experiment_name", "baseline_detector")
        score_threshold = float(self.config.get("evaluation", {}).get("score_threshold", 0.05))

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

        metric_payload = compute_detection_metrics(
            predictions=predictions,
            targets=targets,
            class_names=data_meta["class_names"],
            score_threshold=score_threshold,
        )

        summary = dict(metric_payload["summary"])
        summary.update(
            {
                "experiment_name": experiment_name,
                "split": resolved_split,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                "class_names": list(data_meta["class_names"]),
            }
        )

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

            logger.info("Saved evaluation summary to %s", summary_path)

        return {
            "summary": summary,
            "per_class": metric_payload["per_class"],
        }
