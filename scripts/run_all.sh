#!/usr/bin/env bash
set -euo pipefail

echo "==> Preparing datasets"
python scripts/prepare_large_scale.py
python scripts/prepare_vnwoodknot.py

echo "==> Training baseline"
python scripts/train.py --config configs/train_baseline.yaml

echo "==> Evaluating"
python scripts/evaluate.py --config configs/eval.yaml

echo "==> Exporting results"
python scripts/export_results.py
