#!/usr/bin/env python3
"""Training entry point."""

import argparse

from src.models.hybrid_detector import HybridDetector
from src.engine.trainer import Trainer
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
    model = HybridDetector(
        num_classes=model_cfg.get("num_classes", 10),
        use_transformer=model_cfg.get("use_transformer", False),
        num_transformer_blocks=model_cfg.get("num_transformer_blocks", 0),
        use_p2_branch=model_cfg.get("use_p2_branch", False),
    )

    trainer = Trainer(model=model, config=config)
    logger.info("Starting training...")
    trainer.fit()


if __name__ == "__main__":
    main()
