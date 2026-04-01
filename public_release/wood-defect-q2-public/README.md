# Wood Defect Q2

Codebase for wood surface defect detection experiments on a large-scale in-domain dataset and the VNWoodKnot target-domain benchmark.

## Overview
- Main dataset: large-scale wood surface defect images
- Secondary dataset: VNWoodKnot
- Tasks covered in this repository:
  - dataset preprocessing and manifest generation
  - tiled benchmark construction
  - baseline and hybrid detector training
  - YOLO-based training and evaluation
  - cross-dataset evaluation and efficiency profiling

## Repository Layout
- `src/`: dataset, model, loss, metric, and engine modules
- `scripts/`: entry points for preprocessing, training, evaluation, and profiling
- `configs/`: configuration files for datasets, preprocessing, training, and evaluation
- `tests/`: unit and integration checks for the core pipeline
- `docs/`: concise implementation notes for data preparation

## Data and Artifacts
- Raw datasets are not stored in this repository.
- Processed datasets are expected to live outside the repository and are referenced through config paths.
- Large model checkpoints, logs, and experiment outputs are intentionally excluded from this public copy.

## Environment
- Python dependencies are listed in `requirements.txt`.
- A Conda environment specification is available in `environment.yml`.
- Training and evaluation were designed for a Linux GPU server environment.

## Entry Points
- Preprocessing: `scripts/preprocess_main_for_server.py`, `scripts/preprocess_vnwoodknot_for_server.py`
- Benchmark construction: `scripts/build_screened_benchmark.py`, `scripts/build_yolo_dataset.py`
- Training: `scripts/train.py`, `scripts/train_yolov8.py`
- Evaluation: `scripts/evaluate.py`, `scripts/evaluate_yolov8.py`
- Efficiency profiling: `scripts/profile_efficiency.py`

## Reproducibility Notes
- Split assignment is defined at the source-image level before tiling.
- Configuration files are kept in version control.
- Tests cover the benchmark-construction, preprocessing, export, and evaluation paths used by the repository.
