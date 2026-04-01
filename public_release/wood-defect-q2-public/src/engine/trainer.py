"""Minimal trainer for the baseline detector pipeline."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from src.datasets.manifest_detection_dataset import build_detection_dataloader
from src.engine.evaluator import Evaluator
from src.losses.detection_loss import compute_detection_loss, detach_loss_dict
from src.utils.io import ensure_dir, save_csv, save_json
from src.utils.logger import setup_logger

logger = setup_logger()


class Trainer:
    """Train a manifest-backed baseline detector without a large training framework."""

    def __init__(self, model: Any, config: dict) -> None:
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.get("device", "cpu"))
        self.output_dir = Path(config.get("output_dir", "outputs"))
        self.experiment_name = config.get("experiment_name", "baseline_detector")
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints" / self.experiment_name)
        self.tables_dir = ensure_dir(self.output_dir / "tables")
        self.train_cfg = config.get("train", {})
        self.dataset_cfg = config.get("dataset", {})
        self.split_cfg = config.get("dataset_split", {})
        self.history: list[Dict[str, Any]] = []

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _move_targets_to_device(self, targets):
        moved_targets = []
        for target in targets:
            moved_targets.append(
                {
                    key: value.to(self.device) if hasattr(value, "to") else value
                    for key, value in target.items()
                }
            )
        return moved_targets

    def _build_loader(
        self,
        split_name: str,
        dataset_source: Any,
        shuffle: bool,
        max_samples: int | None = None,
        sampler_config: Dict[str, Any] | None = None,
    ):
        return build_detection_dataloader(
            dataset_config_or_path=dataset_source,
            split=split_name,
            batch_size=int(self.train_cfg.get("batch_size", 2)),
            num_workers=int(self.train_cfg.get("num_workers", 0)),
            shuffle=shuffle,
            split_seed=int(self.split_cfg.get("seed", self.config.get("seed", 42))),
            train_ratio=float(self.split_cfg.get("train_ratio", 0.8)),
            val_ratio=float(self.split_cfg.get("val_ratio", 0.2)),
            max_samples=max_samples,
            sampler_config=sampler_config,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(self.train_cfg.get("weight_decay", 1e-4)),
        )

    def _save_checkpoint(
        self,
        checkpoint_name: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        best_metric: float,
        class_names: list[str],
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config,
                "best_metric": best_metric,
                "class_names": class_names,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def fit(self) -> None:
        """Run the baseline detector training loop."""
        train_source = self.dataset_cfg.get("train")
        val_source = self.dataset_cfg.get("val", train_source)
        if train_source is None:
            raise ValueError("Training config requires dataset.train.")

        max_train_samples = self.train_cfg.get("max_train_samples")
        max_val_samples = self.train_cfg.get("max_val_samples")
        train_split_name = self.dataset_cfg.get("train_split", "train")
        val_split_name = self.dataset_cfg.get("val_split", "val")

        train_loader, train_meta = self._build_loader(
            split_name=train_split_name,
            dataset_source=train_source,
            shuffle=True,
            max_samples=int(max_train_samples) if max_train_samples is not None else None,
            sampler_config=self.train_cfg.get("small_defect_sampler"),
        )
        val_loader, val_meta = self._build_loader(
            split_name=val_split_name,
            dataset_source=val_source,
            shuffle=False,
            max_samples=int(max_val_samples) if max_val_samples is not None else None,
        )

        class_names = list(val_meta.get("class_names") or train_meta.get("class_names", []))
        optimizer = self._build_optimizer()
        evaluator = Evaluator(model=self.model, config=self.config)

        self.model.to(self.device)
        epochs = int(self.train_cfg.get("epochs", 1))
        best_metric_name = str(self.train_cfg.get("best_metric", "mAP50_95"))
        best_metric_value = float("-inf")
        best_val_payload: Dict[str, Any] | None = None

        logger.info(
            "Training %s for %d epochs on %d train images and %d val images",
            self.experiment_name,
            epochs,
            train_meta["num_images"],
            val_meta["num_images"],
        )
        if train_meta.get("sampler"):
            logger.info("Train sampler: %s", train_meta["sampler"])

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_totals: dict[str, float] = defaultdict(float)
            num_batches = 0

            for images, targets, _ in train_loader:
                images = [image.to(self.device) for image in images]
                targets = self._move_targets_to_device(targets)

                optimizer.zero_grad(set_to_none=True)
                loss_dict = self.model(images, targets)
                reduced_loss_dict = compute_detection_loss(loss_dict)
                reduced_loss_dict["loss_total"].backward()
                optimizer.step()

                detached = detach_loss_dict(reduced_loss_dict)
                for key, value in detached.items():
                    running_totals[key] += value
                num_batches += 1

            averaged_losses = {
                f"train_{key}": round(value / max(num_batches, 1), 6)
                for key, value in running_totals.items()
            }

            val_payload = evaluator.evaluate(
                data_loader=val_loader,
                data_meta=val_meta,
                split_name=val_split_name,
                checkpoint_path=None,
                experiment_name=self.experiment_name,
                save_outputs=False,
            )
            val_summary = {
                f"val_{key}": value
                for key, value in val_payload["summary"].items()
                if isinstance(value, (int, float))
            }

            history_row = {"epoch": epoch, **averaged_losses, **val_summary}
            self.history.append(history_row)
            history_df = pd.DataFrame(self.history)
            save_csv(history_df, self.tables_dir / f"{self.experiment_name}_train_history.csv")

            tracked_metric = float(val_payload["summary"].get(best_metric_name, 0.0))
            is_best = tracked_metric >= best_metric_value
            if is_best:
                best_metric_value = tracked_metric
                best_val_payload = val_payload
                self._save_checkpoint("best.pt", epoch, optimizer, best_metric_value, class_names)
                save_json(
                    val_payload["summary"],
                    self.tables_dir / f"{self.experiment_name}_best_val_summary.json",
                )
                save_csv(
                    val_payload["per_class"],
                    self.tables_dir / f"{self.experiment_name}_best_val_per_class.csv",
                )

            self._save_checkpoint("last.pt", epoch, optimizer, best_metric_value, class_names)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_mAP50=%.4f | val_mAP50_95=%.4f | val_preds=%d",
                epoch,
                epochs,
                averaged_losses.get("train_loss_total", 0.0),
                float(val_payload["summary"].get("mAP50", 0.0)),
                float(val_payload["summary"].get("mAP50_95", 0.0)),
                int(val_payload["summary"].get("num_predictions", 0)),
            )

        training_summary = {
            "experiment_name": self.experiment_name,
            "epochs": epochs,
            "best_metric_name": best_metric_name,
            "best_metric_value": round(best_metric_value, 6) if best_metric_value != float("-inf") else None,
            "num_train_images": train_meta["num_images"],
            "num_val_images": val_meta["num_images"],
            "class_names": class_names,
        }
        if train_meta.get("sampler") is not None:
            training_summary["train_sampler"] = train_meta["sampler"]
        if best_val_payload is not None:
            training_summary["best_val_summary"] = best_val_payload["summary"]

        save_json(training_summary, self.tables_dir / f"{self.experiment_name}_train_summary.json")
