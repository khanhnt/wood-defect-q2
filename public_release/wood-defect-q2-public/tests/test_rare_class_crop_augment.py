"""Checks for offline rare-class crop augmentation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.datasets.rare_class_crop_augment import build_rare_class_crop_augmented_dataset


def _make_annotation(class_name, class_id, bbox_xyxy_norm):
    x1, y1, x2, y2 = bbox_xyxy_norm
    return {
        "class_name": class_name,
        "class_id": class_id,
        "bbox_xyxy_norm": [x1, y1, x2, y2],
        "bbox_width_norm": round(x2 - x1, 6),
        "bbox_height_norm": round(y2 - y1, 6),
        "bbox_area_norm": round((x2 - x1) * (y2 - y1), 6),
        "is_small_defect": False,
        "source_label": class_name,
    }


def _make_record(image_path, annotations, split="train", source_image_id="src_0"):
    return {
        "dataset_name": "toy",
        "image_id": "images/train/tile_0",
        "image_path": str(image_path),
        "split": split,
        "source_image_id": source_image_id,
        "width": 1024,
        "height": 1024,
        "annotations": annotations,
        "is_empty": len(annotations) == 0,
        "empty_reason": None if annotations else "negative_tile",
        "issues": [],
        "num_invalid_boxes": 0,
        "num_clipped_boxes": 0,
        "annotation_path": None,
        "semantic_map_path": None,
        "tile_origin_xy": [0, 0],
        "tile_index": 0,
        "num_small_annotations": 0,
    }


def test_build_rare_class_crop_augmented_dataset_generates_valid_crop(tmp_path):
    image_root_dir = tmp_path / "images_root"
    output_root_dir = tmp_path / "augmented"
    image_root_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_root_dir / "images" / "train" / "tile_0.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1024, 1024), color=(180, 180, 180)).save(image_path)

    record = _make_record(
        image_path="images/train/tile_0.jpg",
        annotations=[
            _make_annotation("resin", 0, [0.45, 0.45, 0.50, 0.50]),
            _make_annotation("dead_knot", 1, [0.52, 0.46, 0.60, 0.58]),
        ],
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = build_rare_class_crop_augmented_dataset(
        input_manifest_path=manifest_path,
        image_root_dir=image_root_dir,
        output_root_dir=output_root_dir,
        dataset_name="toy_aug",
        target_classes=["resin"],
        class_max_crops={"resin": 1},
        repo_output_dir=tmp_path / "tables",
    )

    output_manifest = output_root_dir / "manifest.jsonl"
    assert output_manifest.exists()
    lines = output_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    original_record = json.loads(lines[0])
    augmented_record = json.loads(lines[1])
    assert original_record["image_path"] == str(image_path)
    assert augmented_record["augmentation_type"] == "rare_class_crop"
    assert augmented_record["augmentation_primary_class"] == "resin"
    assert augmented_record["parent_image_id"] == "images/train/tile_0"
    assert augmented_record["source_image_id"] == "src_0"
    assert augmented_record["width"] >= 320
    assert augmented_record["height"] == augmented_record["width"]
    assert len(augmented_record["annotations"]) >= 1
    assert any(annotation["class_name"] == "resin" for annotation in augmented_record["annotations"])
    for annotation in augmented_record["annotations"]:
        x1, y1, x2, y2 = annotation["bbox_xyxy_norm"]
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0

    assert Path(augmented_record["image_path"]).exists()
    assert result["summary"]["num_augmented_records"] == 1


def test_build_rare_class_crop_augmented_dataset_rejects_edge_targets(tmp_path):
    image_root_dir = tmp_path / "images_root"
    output_root_dir = tmp_path / "augmented"
    image_root_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_root_dir / "images" / "train" / "tile_0.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1024, 1024), color=(180, 180, 180)).save(image_path)

    record = _make_record(
        image_path="images/train/tile_0.jpg",
        annotations=[_make_annotation("knot_missing", 0, [0.0, 0.40, 0.03, 0.45])],
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = build_rare_class_crop_augmented_dataset(
        input_manifest_path=manifest_path,
        image_root_dir=image_root_dir,
        output_root_dir=output_root_dir,
        dataset_name="toy_aug",
        target_classes=["knot_missing"],
        repo_output_dir=tmp_path / "tables",
    )

    output_manifest = output_root_dir / "manifest.jsonl"
    lines = output_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert result["summary"]["num_augmented_records"] == 0
    assert result["summary"]["augmentation"]["rejection_counts"]["target_near_tile_edge"] >= 1


def test_build_rare_class_crop_augmented_dataset_balanced_mode_prefers_lower_head_context(tmp_path):
    image_root_dir = tmp_path / "images_root"
    output_root_dir = tmp_path / "augmented"
    image_root_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for stem in ("tile_a", "tile_b"):
        image_path = image_root_dir / "images" / "train" / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1024, 1024), color=(180, 180, 180)).save(image_path)
        image_paths.append(image_path)

    record_a = {
        **_make_record(
            image_path="images/train/tile_a.jpg",
            annotations=[
                _make_annotation("resin", 0, [0.45, 0.45, 0.50, 0.50]),
                _make_annotation("live_knot", 1, [0.52, 0.46, 0.60, 0.58]),
            ],
            source_image_id="src_a",
        ),
        "image_id": "images/train/tile_a",
    }
    record_b = {
        **_make_record(
            image_path="images/train/tile_b.jpg",
            annotations=[_make_annotation("resin", 0, [0.45, 0.45, 0.50, 0.50])],
            source_image_id="src_b",
        ),
        "image_id": "images/train/tile_b",
    }
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(record_a) + "\n" + json.dumps(record_b) + "\n",
        encoding="utf-8",
    )

    result = build_rare_class_crop_augmented_dataset(
        input_manifest_path=manifest_path,
        image_root_dir=image_root_dir,
        output_root_dir=output_root_dir,
        dataset_name="toy_aug",
        target_classes=["resin"],
        class_max_crops={"resin": 1},
        candidate_selection_mode="balanced",
        head_classes=["live_knot"],
        repo_output_dir=tmp_path / "tables",
    )

    output_manifest = output_root_dir / "manifest.jsonl"
    lines = output_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    augmented_record = json.loads(lines[-1])
    assert augmented_record["parent_image_id"] == "images/train/tile_b"
    assert result["summary"]["augmentation"]["generated_by_class"]["resin"] == 1
