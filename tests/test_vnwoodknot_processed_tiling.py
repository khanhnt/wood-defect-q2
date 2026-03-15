from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.datasets.base_dataset import build_annotation
from src.datasets.vnwoodknot_dataset import build_tiled_vnwoodknot_from_processed_manifest


def test_build_tiled_vnwoodknot_from_processed_manifest(tmp_path):
    processed_root = tmp_path / "vnwoodknot_processed"
    image_dir = processed_root / "images" / "test" / "live_knot"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (1500, 1500), color=(120, 110, 100)).save(image_path)

    manifest_path = processed_root / "manifest.jsonl"
    record = {
        "dataset_name": "vnwoodknot",
        "image_id": "images/test/live_knot/sample",
        "image_path": "images/test/live_knot/sample.jpg",
        "split": "test",
        "source_category": "live_knot",
        "width": 1500,
        "height": 1500,
        "annotations": [
            build_annotation(
                class_name="live_knot",
                bbox_xyxy_norm=[0.4, 0.4, 0.6, 0.6],
                source_label="0",
            )
        ],
        "is_empty": False,
        "empty_reason": None,
        "issues": [],
        "num_invalid_boxes": 0,
        "num_clipped_boxes": 0,
        "annotation_path": None,
        "semantic_map_path": None,
    }
    manifest_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    output_root = tmp_path / "vnwoodknot_tiled"
    result = build_tiled_vnwoodknot_from_processed_manifest(
        input_manifest_path=manifest_path,
        image_root_dir=processed_root,
        output_root_dir=output_root,
        dataset_name="vnwoodknot_tiled",
    )

    processed_records = result["processed_records"]
    assert len(processed_records) == 4
    assert all(record["split"] == "test" for record in processed_records)
    assert all(record["width"] == 1024 for record in processed_records)
    assert all(record["height"] == 1024 for record in processed_records)
    assert all("tile_origin_xy" in record for record in processed_records)
    assert all(record["source_image_id"] == "images/test/live_knot/sample" for record in processed_records)
    assert all(len(record["annotations"]) == 1 for record in processed_records)

    output_manifest = output_root / "manifest.jsonl"
    assert output_manifest.exists()
    output_lines = output_manifest.read_text(encoding="utf-8").splitlines()
    assert len(output_lines) == 4
