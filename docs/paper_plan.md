# Paper Plan

## Provisional title
A Lightweight Hybrid CNN-Transformer Detector for Robust Wood Defect Detection with Improved Small-Defect Sensitivity

## Problem statement
We study wood defect detection under three priorities:
1. improved detection of small defects
2. computational efficiency
3. cross-dataset robustness

## Datasets
- Main dataset: Large-scale wood surface defects
- Secondary dataset: VNWoodKnot

## Small-defect definition
We use one transparent rule across dataset audits and experiments.

A defect is tagged as small when its bounding box satisfies the configured small-defect rule:
- `bbox_area_ratio <= 0.01`
- or `bbox_width_px <= 16`
- or `bbox_height_px <= 16`

The default combine mode is `any`, so meeting any one of the three conditions is sufficient.
These thresholds are stored in the dataset config under `small_defect` so the definition remains fixed and reproducible across audits and training runs.

## Planned experiments
- E0: Baseline detector on main dataset
- E1: Baseline + P2 small-defect branch
- E2: Baseline + Transformer blocks
- E3: Baseline + Transformer + P2 branch
- E4: Efficiency benchmark
- E5: Robustness benchmark
- E6: Cross-dataset evaluation on VNWoodKnot
- E7: Fine-tuning / adaptation on VNWoodKnot

## Notes
The preprocessing and processed-dataset structure used for server upload are documented in `docs/preprocessing_for_server.md`.

This file will be updated during development.
