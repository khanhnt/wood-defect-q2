#!/usr/bin/env python3
"""Evaluation entry point for the simple baseline detector pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.evaluator import Evaluator
from src.models.heads.detection_head import build_baseline_detector
from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override")
    return parser.parse_args()


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))

    experiment_name = config.get("experiment_name", "baseline_detector")
    output_dir = Path(config.get("output_dir", "outputs"))
    checkpoint_path = Path(
        args.checkpoint
        or config.get("checkpoint_path")
        or (output_dir / "checkpoints" / experiment_name / "best.pt")
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint.get("config", {})
    config = _deep_merge_dict(train_config, config)
    config["checkpoint_path"] = str(checkpoint_path)

    model_cfg = dict(config.get("model", {}))
    if "num_classes" not in model_cfg:
        model_cfg["num_classes"] = len(checkpoint.get("class_names", []))

    model_name = model_cfg.get("name", "baseline_detector")
    if model_name != "baseline_detector":
        raise NotImplementedError(
            f"Only the baseline detector path is implemented now. Received model.name={model_name!r}."
        )

    model = build_baseline_detector(model_config=model_cfg, train_config=config.get("train", {}))
    evaluator = Evaluator(model=model, config=config)
    logger.info("Starting evaluation for %s using %s", experiment_name, checkpoint_path)
    metrics = evaluator.evaluate(
        checkpoint_path=checkpoint_path,
        experiment_name=experiment_name,
        save_outputs=True,
    )
    logger.info("Evaluation results: %s", metrics["summary"])


if __name__ == "__main__":
    main()
