# Data Preparation for Server Training

## Goal
Build compact object-detection datasets outside the git repository so they can be uploaded to a Linux GPU server without carrying the raw BMP trees.

## Rules
- Raw datasets stay outside the repo.
- Processed datasets also stay outside the repo.
- Only code, configs, logs, docs, and small summary outputs stay inside the repo.
- Semantic maps are ignored.
- The processed format is manifest-based and portable across macOS and Linux.

## Main Dataset Output
Processed root example:

```text
processed_root/
  images/
    train/
      Images1/
        image_001__x0000_y0000_w1024_h1024.jpg
    val/
    test/
  manifest.jsonl
  metadata.json
```

Notes:
- Source BMP images are converted to JPEG with `quality=97`.
- Tiling is done before training.
- Each tile keeps remapped bounding boxes in tile coordinates.
- Positive tiles are always kept.
- Negative tiles are sampled with a reproducible policy from config.
- Split assignment is done at the source-image level before tiling to avoid leakage between train and validation tiles.

## VNWoodKnot Output
Processed root example:

```text
processed_root/
  images/
    train/
      live_knot/
        img_0001.jpg
    val/
      dead_knot/
        img_0002.jpg
    test/
      knot_free/
        img_0003.jpg
  manifest.jsonl
  metadata.json
```

Notes:
- Images are re-saved as JPEG with `quality=97`.
- Original dataset splits are preserved.
- Labels are stored in the same unified manifest format used by the main dataset.

## Manifest Fields
Each JSONL record contains:
- `dataset_name`
- `image_id`
- `image_path`
- `split`
- `source_category`
- `source_image_id`
- `width`
- `height`
- `annotations`
- `is_empty`
- `empty_reason`

Important portability detail:
- `image_path` is stored relative to the processed dataset root.
- Training configs should set `root_dir` to the processed dataset root and `manifest_path` to `processed_root/manifest.jsonl`.

## Repo Outputs
Each preprocessing run also writes compact summaries inside the repo:
- `outputs/tables/<dataset>_preprocess_summary.json`
- `outputs/tables/<dataset>_preprocess_class_distribution.csv`

These files are small and intended for quick inspection and reproducibility checks.
