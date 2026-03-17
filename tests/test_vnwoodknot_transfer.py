from __future__ import annotations

import json

from PIL import Image

from src.datasets.base_dataset import build_annotation
from src.datasets.vnwoodknot_transfer import build_vnwoodknot_transfer_yolo_dataset


def test_build_vnwoodknot_transfer_yolo_dataset_keeps_negative_knot_free_images(tmp_path):
    processed_root = tmp_path / "vnwoodknot_processed"
    live_dir = processed_root / "images" / "train" / "live_knot"
    free_dir = processed_root / "images" / "train" / "knot_free"
    live_dir.mkdir(parents=True, exist_ok=True)
    free_dir.mkdir(parents=True, exist_ok=True)

    live_path = live_dir / "live.jpg"
    free_path = free_dir / "free.jpg"
    Image.new("RGB", (1200, 1200), color=(120, 110, 100)).save(live_path)
    Image.new("RGB", (1200, 1200), color=(150, 140, 130)).save(free_path)

    manifest_path = processed_root / "manifest.jsonl"
    records = [
        {
            "dataset_name": "vnwoodknot",
            "image_id": "images/train/live_knot/live",
            "image_path": "images/train/live_knot/live.jpg",
            "split": "train",
            "source_image_id": "images/train/live_knot/live",
            "width": 1200,
            "height": 1200,
            "annotations": [
                build_annotation(
                    class_name="live_knot",
                    bbox_xyxy_norm=[0.2, 0.2, 0.5, 0.5],
                    source_label="1",
                )
            ],
            "is_empty": False,
            "empty_reason": None,
        },
        {
            "dataset_name": "vnwoodknot",
            "image_id": "images/train/knot_free/free",
            "image_path": "images/train/knot_free/free.jpg",
            "split": "train",
            "source_image_id": "images/train/knot_free/free",
            "width": 1200,
            "height": 1200,
            "annotations": [
                build_annotation(
                    class_name="knot_free",
                    bbox_xyxy_norm=[0.1, 0.1, 0.2, 0.2],
                    source_label="0",
                )
            ],
            "is_empty": False,
            "empty_reason": None,
        },
    ]
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    result = build_vnwoodknot_transfer_yolo_dataset(
        input_manifest_path=manifest_path,
        image_root_dir=processed_root,
        manifest_output_root_dir=processed_root / "benchmarks" / "vnwoodknot_live_dead_2class",
        yolo_output_root_dir=processed_root / "benchmarks" / "vnwoodknot_live_dead_2class_yolo",
    )

    filtered_summary = result["filtered_manifest"]["summary"]
    yolo_summary = result["yolo_export"]["summary"]

    assert filtered_summary["kept_classes"] == ["live_knot", "dead_knot"]
    assert filtered_summary["num_selected_source_images"] == 2
    assert filtered_summary["selected_annotation_count_by_class"]["live_knot"] == 1
    assert filtered_summary["selected_annotation_count_by_class"]["dead_knot"] == 0

    assert yolo_summary["num_records_by_split"]["train"] == 2
    assert yolo_summary["positive_records_by_split"]["train"] == 1
    assert yolo_summary["negative_records_by_split"]["train"] == 1
    assert yolo_summary["classes"] == ["live_knot", "dead_knot"]
