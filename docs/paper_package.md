# Paper Package Notes

## Trade-off Table Plan
- **Rows:** `YOLOv8s@single-gpu`, `YOLOv8s@4xGPU`, `YOLOv8m@single-gpu`, `YOLOv8m@4xGPU`
- **Columns:** params, GFLOPs, inference latency (ms/img @ 1024), trained batch, validation mAP50, mAP50_95, small-target mAP50_95, GPU-hours per run, checkpoint size.
- Include explicit note that `YOLOv8m` serves as a *scaling control* with similar inference recipe but higher compute, not the proposed method.

## Qualitative Figure Plan
- **Figure 1:** Mosaic of 6 in-domain tiles grouped by defect type (`live_knot`, `dead_knot`, `resin`, `crack/knot_with_crack`, `marrow`, `knot_missing`) showing detection box confidence changes when moving from YOLOv8s to YOLOv8m.
- **Figure 2:** Side-by-side crop of a VNWoodKnot image tile with YOLOv8s predictions (mapped classes) and YOLOv8s predictions on the matched-tile VNWoodKnot tiling (Y6a), annotated with FP/false labeling.
- **Caption focus:** highlight failure modes (missed small cracks, overconfident live knot vs resin) and show YOLOv8m gives negligible visual improvement despite heavier architecture.

## Benchmark Reconstruction Outline
1. **Dataset recapitulation:** describe original VSB subset (7 classes, 3600 screened images) and link to builder scripts for transparency.
2. **Manifest pipeline:** summarize how `scripts/build_screened_benchmark.py` reruns the selection logic; highlight available config values (`selection_mode`, `target_source_images`).
3. **Protocol contrast:** list differences between `Y0-3600` (screened) and `Y0-full_7class` (full data 7-class) in terms of source image count, class coverage, annotation density.
4. **Reproducibility commitments:** mention that both manifest builders, YOLO exports, and evaluation configs are checked in; refer to `docs/README.md` (if exists) or create link to `configs/*` for dataset definitions.

## External Section (Severe Transfer Gap Narrative)
- **Opening claim:** the zero-shot transfer from VSB to VNWoodKnot yields `mAP50_95 < 0.02` despite matching label subsets, demonstrating a severe semantic/scale gap.
- **Evidence:** cite `Y6` failure (~0.014) and `Y6a` matched-tiling failure (~0.006) to show even protocol alignment does not recover accuracy.
- **Interpretation:** frame as “domain gap + taxonomy mismatch”; mention `ignored_prediction_count` statistics and the fact that most predictions map to non-overlap classes or are filtered out when requiring live/dead knot.
- **No robustness claim:** explicitly state we do not claim general robustness; instead position this as an empirical diagnosis that motivates future work (e.g., class-agnostic transfers or fine-tuning). 
 
## Artifact Checklist (Repo/Public)
- `configs/dataset_main_vsb7_3600_rarefirst.yaml` and `configs/dataset_main_full_7class.yaml`: dataset definitions for in-domain eval.
- `scripts/build_screened_benchmark.py`: reproduces screened manifest selection.
- `scripts/build_yolo_dataset.py`: YOLO export helper.
- `scripts/train_yolov8.py` + `scripts/evaluate_yolov8.py`: training/eval entrypoints reusable by others.
- `scripts/build_rare_class_crop_augment.py`: optional augmentation build for rare-defect study.
- `scripts/preprocess_vnwoodknot_for_server.py` & `scripts/build_tiled_vnwoodknot_from_processed.py`: preprocessing/export for external dataset.
- `outputs/tables/*_summary.json` + per-class/per-split CSVs for Y0, Y1, Y2, Y6, Y6a, Y0-3600, Y0-full, and YOLOv8m: include these in supplementary data.
- `configs/dataset_transfer_tiled.yaml`: label mapping used for external evaluation.
