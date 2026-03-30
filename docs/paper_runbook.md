# Paper Runbook

This runbook turns the locked experiment matrix into repeatable commands for the remaining seeds.

Assumption:

- the currently pulled single-run baselines are the default-seed runs (`seed=42` in `scripts/train_yolov8.py`)
- if you want a fully fresh 3-seed pack, rerun `42` as well; otherwise only add `43` and `44`

## Environment

```bash
export WOOD_MAIN_PROCESSED_ROOT='/storage/tonlh/khanhnt/2026/processed_for_server/main_dataset'
```

## 1. `Y0-full_7class` (remaining seeds to reach 3)

Branch:

```bash
git checkout codex/y0-yolov8-model-switch
git pull
```

Per seed:

```bash
export SEED=43
export CUDA_VISIBLE_DEVICES=0

nohup bash -lc "
python scripts/train_yolov8.py \
  --data \"${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/full_7class_yolo/dataset.yaml\" \
  --model yolov8s \
  --experiment-name \"y0_yolov8s_full_7class_seed${SEED}\" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 32 \
  --device 0 \
  --seed \"${SEED}\" \
  --workers 4

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_full_7class.yaml \
  --checkpoint \"outputs/yolo/y0_yolov8s_full_7class_seed${SEED}/weights/best.pt\" \
  --experiment-name \"y0_yolov8s_full_7class_seed${SEED}_eval\" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
" > "y0_yolov8s_full_7class_seed${SEED}.log" 2>&1 &
```

Run for `SEED=43`, `44`.

## 2. `Y0-3600` (remaining seeds to reach 3)

Branch:

```bash
git checkout codex/y0-yolov8-model-switch
git pull
```

Per seed:

```bash
export SEED=43
export CUDA_VISIBLE_DEVICES=0

nohup bash -lc "
python scripts/train_yolov8.py \
  --data \"${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml\" \
  --model yolov8s \
  --experiment-name \"y0_yolov8s_vsb7_3600_rarefirst_seed${SEED}\" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 32 \
  --device 0 \
  --seed \"${SEED}\" \
  --workers 4

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_vsb7_3600_rarefirst.yaml \
  --checkpoint \"outputs/yolo/y0_yolov8s_vsb7_3600_rarefirst_seed${SEED}/weights/best.pt\" \
  --experiment-name \"y0_yolov8s_vsb7_3600_rarefirst_seed${SEED}_eval\" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
" > "y0_yolov8s_vsb7_3600_rarefirst_seed${SEED}.log" 2>&1 &
```

Run for `SEED=43`, `44`.

## 3. `Y1` (2 extra seeds)

Branch:

```bash
git checkout codex/y1-yolov8-p2
git pull
```

Per seed:

```bash
export SEED=43
export CUDA_VISIBLE_DEVICES=0

nohup bash -lc "
python scripts/train_yolov8.py \
  --data \"${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml\" \
  --model-config configs/models/yolov8s-p2-7class.yaml \
  --weights yolov8s.pt \
  --experiment-name \"y1_yolov8s_p2_vsb7_3600_rarefirst_seed${SEED}\" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 16 \
  --device 0 \
  --seed \"${SEED}\" \
  --workers 4

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_vsb7_3600_rarefirst.yaml \
  --checkpoint \"outputs/yolo/y1_yolov8s_p2_vsb7_3600_rarefirst_seed${SEED}/weights/best.pt\" \
  --experiment-name \"y1_yolov8s_p2_vsb7_3600_rarefirst_seed${SEED}_eval\" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
" > "y1_yolov8s_p2_vsb7_3600_rarefirst_seed${SEED}.log" 2>&1 &
```

Run for `SEED=43`, `44`.

## 4. `Y2` (2 extra seeds)

Branch:

```bash
git checkout codex/y2-yolov8-wniou
git pull
```

Per seed:

```bash
export SEED=43
export CUDA_VISIBLE_DEVICES=0

nohup bash -lc "
python scripts/train_yolov8.py \
  --data \"${WOOD_MAIN_PROCESSED_ROOT}/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml\" \
  --model yolov8s \
  --experiment-name \"y2_yolov8s_wniou_vsb7_3600_rarefirst_seed${SEED}\" \
  --epochs 20 \
  --imgsz 1024 \
  --batch 16 \
  --device 0 \
  --seed \"${SEED}\" \
  --workers 4 \
  --box-loss wniou \
  --wniou-lambda-nwd 0.30 \
  --wniou-focus-alpha 0.50 \
  --wniou-focus-gamma 1.00 \
  --wniou-distance-scale 0.05

python scripts/evaluate_yolov8.py \
  --dataset-config configs/dataset_main_vsb7_3600_rarefirst.yaml \
  --checkpoint \"outputs/yolo/y2_yolov8s_wniou_vsb7_3600_rarefirst_seed${SEED}/weights/best.pt\" \
  --experiment-name \"y2_yolov8s_wniou_vsb7_3600_rarefirst_seed${SEED}_eval\" \
  --split val \
  --batch 32 \
  --imgsz 1024 \
  --device 0 \
  --score-threshold 0.05 \
  --iou-threshold 0.5 \
  --max-detections 100 \
  --small-defect-eval
" > "y2_yolov8s_wniou_vsb7_3600_rarefirst_seed${SEED}.log" 2>&1 &
```

Run for `SEED=43`, `44`.

## 5. Optional scaling-control rerun

No additional rerun is required for `Y0m-3600` unless we need a second control seed for rebuttal or supplementary material. The current pulled run is enough to support the statement that simple model scaling does not materially improve the benchmark.

## 6. Aggregation Checklist

After each run, collect:

- `outputs/tables/<experiment>_train_summary.json`
- `outputs/tables/<experiment>_eval_val_summary.json`

Then update:

- `outputs/tables/paper_experiment_matrix_current.csv`
- `docs/paper_experiment_matrix_current.csv`
- the main paper table with mean ± std for `Y0-full_7class`, `Y0-3600`, `Y1`, and `Y2`
