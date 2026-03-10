#!/usr/bin/env python3
"""Training entry point for the simple baseline detector pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.trainer import Trainer
from src.models.heads.detection_head import build_baseline_detector
from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "baseline_detector")
    if model_name != "baseline_detector":
        raise NotImplementedError(
            f"Only the baseline detector path is implemented now. Received model.name={model_name!r}."
        )

    model = build_baseline_detector(model_config=model_cfg, train_config=config.get("train", {}))
    trainer = Trainer(model=model, config=config)
    logger.info("Starting training for %s", config.get("experiment_name", "baseline_detector"))
    trainer.fit()


if __name__ == "__main__":
    main()
