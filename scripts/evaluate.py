#!/usr/bin/env python3
"""Evaluation entry point."""

import argparse

from src.models.hybrid_detector import HybridDetector
from src.engine.evaluator import Evaluator
from src.utils.config import load_yaml
from src.utils.logger import setup_logger

logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to evaluation config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    model = HybridDetector()
    evaluator = Evaluator(model=model, config=config)
    logger.info("Starting evaluation...")
    metrics = evaluator.evaluate()
    logger.info("Evaluation results: %s", metrics)


if __name__ == "__main__":
    main()
