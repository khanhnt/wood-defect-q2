from __future__ import annotations

import json

from src.datasets.class_filtered_manifest import build_class_filtered_manifest


def _make_annotation(class_name: str, class_id: int) -> dict:
    return {
        "class_name": class_name,
        "class_id": class_id,
        "bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.2],
        "bbox_width_norm": 0.1,
        "bbox_height_norm": 0.1,
        "bbox_area_norm": 0.01,
        "is_small_defect": False,
        "source_label": class_name,
    }


def _make_record(image_id: str, source_image_id: str, split: str, annotations: list[dict]) -> dict:
    return {
        "dataset_name": "toy",
        "image_id": image_id,
        "image_path": f"images/{split}/{image_id}.jpg",
        "source_image_id": source_image_id,
        "split": split,
        "width": 1024,
        "height": 1024,
        "annotations": annotations,
        "is_empty": len(annotations) == 0,
        "empty_reason": None if annotations else "negative_tile",
        "num_small_annotations": 0,
    }


def test_build_class_filtered_manifest_drops_sources_without_kept_classes(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    records = [
        _make_record("a", "src_a", "train", [_make_annotation("live_knot", 0)]),
        _make_record("b", "src_b", "val", [_make_annotation("blue_stain", 8)]),
        _make_record("c", "src_c", "test", []),
    ]
    manifest_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    result = build_class_filtered_manifest(
        input_manifest_path=manifest_path,
        output_root_dir=tmp_path / "filtered",
        dataset_name="toy_7class",
        kept_classes=["live_knot"],
    )

    output_manifest = tmp_path / "filtered" / "manifest.jsonl"
    lines = output_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    filtered_record = json.loads(lines[0])
    assert filtered_record["source_image_id"] == "src_a"
    assert filtered_record["annotations"][0]["class_id"] == 0
    assert result["summary"]["num_dropped_source_images_without_kept_classes"] == 2


def test_build_class_filtered_manifest_can_keep_empty_sources(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    records = [
        _make_record("a", "src_a", "train", [_make_annotation("live_knot", 0)]),
        _make_record("b", "src_b", "val", [_make_annotation("blue_stain", 8)]),
    ]
    manifest_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    build_class_filtered_manifest(
        input_manifest_path=manifest_path,
        output_root_dir=tmp_path / "filtered",
        dataset_name="toy_7class",
        kept_classes=["live_knot"],
        drop_source_images_without_kept_classes=False,
    )

    output_manifest = tmp_path / "filtered" / "manifest.jsonl"
    lines = output_manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    filtered_records = [json.loads(line) for line in lines]
    rare_only_record = next(record for record in filtered_records if record["source_image_id"] == "src_b")
    assert rare_only_record["annotations"] == []
    assert rare_only_record["is_empty"] is True
    assert rare_only_record["empty_reason"] == "negative_tile_class_filtered"
