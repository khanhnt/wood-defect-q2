# Experiment Matrix

This file locks the paper matrix, records the currently available metrics, and lists the reruns needed to make the paper statistically credible.

## Locked Main Matrix

| ID | Experiment | Role in paper | Metric source | mAP50 | mAP50_95 | Small-target mAP50_95 | Status | Decision |
|---|---|---|---|---:|---:|---:|---|---|
| `Y0-full_7class` | `y0_yolov8s_full_7class` | full-data 7-class in-domain baseline | repo eval | 0.7692 | 0.4432 | 0.1776 | single seed complete | keep; needs 3 seeds |
| `Y0-3600` | `y0_yolov8s_vsb7_3600_rarefirst` | main screened-benchmark baseline | repo eval | 0.8006 | 0.4597 | 0.1875 | single seed complete | keep; needs 3 seeds |
| `Y1` | `y1_yolov8s_p2_vsb7_3600_rarefirst` | small-object head ablation | repo eval | 0.7835 | 0.4467 | 0.1803 | single seed complete | negative result; add 2 seeds |
| `Y2` | `y2_yolov8s_wniou_vsb7_3600_rarefirst` | loss ablation on top of `Y0-3600` | repo eval | 0.7954 | 0.4503 | 0.1818 | single seed complete | negative result; add 2 seeds if resources allow |
| `Y6` | `y6_y0_vnwoodknot_eval` | strict zero-shot external transfer | repo eval | 0.0187 | 0.0139 | N/A | single run complete | keep as severe transfer-gap evidence |
| `Y6a` | `y6a_y0_vnwoodknot_tiled_raw_eval` | matched-tiling zero-shot external transfer | repo eval | 0.0090 | 0.0062 | N/A | single run complete | keep as protocol-alignment negative result |

## Scaling Control

| ID | Experiment | Role in paper | Metric source | mAP50 | mAP50_95 | Small-target mAP50_95 | Inference ms/img | Params | GFLOPs | Decision |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `Y0m-3600` | `y0_yolov8m_vsb7_3600_rarefirst` | detector scaling control | repo eval | 0.7980 | 0.4539 | 0.1892 | 6.7 | 23.2M | 67.4 | keep as control only; do not frame as main method |

Notes:
- `Y0m-3600` now has both repo-eval and Ultralytics-native validation outputs. The native-val line was `0.8222 / 0.4876`, so both metric sources tell the same story.
- The correct reading so far is that scaling from `YOLOv8s` to `YOLOv8m` gives negligible gain relative to the compute increase.

## Seed Plan

### Must run

1. `Y0-full_7class`: add seeds `43`, `44` to the existing default-seed (`42`) run
2. `Y0-3600`: add seeds `43`, `44` to the existing default-seed (`42`) run

### Strongly recommended

1. `Y1`: add seeds `43`, `44`
2. `Y2`: add seeds `43`, `44`

### No additional reruns required

1. `Y6`
2. `Y6a`

The external runs are not being used to claim robustness. They are being used to support the claim of a **severe transfer gap**, so repeated seeds are much lower priority than in-domain runs.

## Shared Environment

```bash
export WOOD_MAIN_PROCESSED_ROOT='/storage/tonlh/khanhnt/2026/processed_for_server/main_dataset'
export CUDA_VISIBLE_DEVICES=0
```

For 4-GPU reruns, switch to:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Command Templates

Branch requirements:

- `Y0-full_7class`, `Y0-3600`, `Y0m-3600`: run from a branch that contains the friendly `--model` selector, e.g. `codex/y0-yolov8-model-switch`.
- `Y1`: run from `codex/y1-yolov8-p2`.
- `Y2`: run from `codex/y2-yolov8-wniou`.

### `Y0-full_7class`

```bash
python scripts/train_yolov8.py \
  --data "${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/full_7class_yolo/dataset.yaml" \
  --model yolov8s \
  --experiment-name "y0_yolov8s_full_7class_seed${SEED}" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 32 \
  --device 0 \
  --seed "${SEED}" \
  --workers 4

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_full_7class.yaml \
  --checkpoint "outputs/yolo/y0_yolov8s_full_7class_seed${SEED}/weights/best.pt" \
  --experiment-name "y0_yolov8s_full_7class_seed${SEED}_eval" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
```

### `Y0-3600`

```bash
python scripts/train_yolov8.py \
  --data "${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml" \
  --model yolov8s \
  --experiment-name "y0_yolov8s_vsb7_3600_rarefirst_seed${SEED}" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 32 \
  --device 0 \
  --seed "${SEED}" \
  --workers 4

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_vsb7_3600_rarefirst.yaml \
  --checkpoint "outputs/yolo/y0_yolov8s_vsb7_3600_rarefirst_seed${SEED}/weights/best.pt" \
  --experiment-name "y0_yolov8s_vsb7_3600_rarefirst_seed${SEED}_eval" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
```

### `Y1`

```bash
python scripts/train_yolov8.py \
  --data "${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml" \
  --weights yolov8s.pt \
  --model-config configs/models/yolov8s-p2-7class.yaml \
  --experiment-name "y1_yolov8s_p2_vsb7_3600_rarefirst_seed${SEED}" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 16 \
  --device 0 \
  --seed "${SEED}" \
  --workers 4

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_vsb7_3600_rarefirst.yaml \
  --checkpoint "outputs/yolo/y1_yolov8s_p2_vsb7_3600_rarefirst_seed${SEED}/weights/best.pt" \
  --experiment-name "y1_yolov8s_p2_vsb7_3600_rarefirst_seed${SEED}_eval" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
```

### `Y2`

```bash
python scripts/train_yolov8.py \
  --data "${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml" \
  --model yolov8s \
  --experiment-name "y2_yolov8s_wniou_vsb7_3600_rarefirst_seed${SEED}" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 16 \
  --device 0 \
  --seed "${SEED}" \
  --workers 4 \
  --box-loss wniou \
  --wniou-lambda-nwd 0.30 \
  --wniou-focus-alpha 0.50 \
  --wniou-focus-gamma 1.00 \
  --wniou-distance-scale 0.05

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_vsb7_3600_rarefirst.yaml \
  --checkpoint "outputs/yolo/y2_yolov8s_wniou_vsb7_3600_rarefirst_seed${SEED}/weights/best.pt" \
  --experiment-name "y2_yolov8s_wniou_vsb7_3600_rarefirst_seed${SEED}_eval" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
```

## Aggregation Rule

After each rerun, collect:

- `outputs/tables/<experiment>_train_summary.json`
- `outputs/tables/<experiment>_eval_val_summary.json`

Then aggregate into one master sheet with:

- `experiment_id`
- `seed`
- `dataset`
- `model_family`
- `metric_source`
- `mAP50`
- `mAP50_95`
- `precision50`
- `recall50`
- `small_target_mAP50_95`
- `num_predictions`

The main paper should report **mean ± std** for `Y0-full_7class`, `Y0-3600`, `Y1`, and `Y2`.
