from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.datasets.vnwoodknot_dataset import preprocess_vnwoodknot_for_server


def test_preprocess_vnwoodknot_for_server_tiled(tmp_path):
    raw_root = tmp_path / "raw"
    image_dir = raw_root / "train" / "1"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (1500, 1500), color=(150, 140, 130)).save(image_path)
    label_path = image_dir / "sample.txt"
    # A centered box that appears in all four overlapping 1024x1024 tiles.
    label_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    processed_root = tmp_path / "processed"
    repo_output_dir = tmp_path / "repo_tables"
    config = {
        "dataset_name": "vnwoodknot_tiled",
        "root_dir": str(raw_root),
        "processed_root_dir": str(processed_root),
        "repo_output_dir": str(repo_output_dir),
        "classes": ["live_knot", "dead_knot", "knot_free"],
        "source_category_map": {"0": "knot_free", "1": "live_knot", "2": "dead_knot"},
        "yolo_class_map": {"0": "live_knot", "1": "dead_knot"},
        "splits": ["train"],
        "split": {"seed": 42, "train_ratio": 1.0, "val_ratio": 0.0, "test_ratio": 0.0},
        "tile": {
            "enabled": True,
            "size": 1024,
            "overlap": 128,
            "min_box_visibility": 0.5,
            "keep_all_negative_tiles": True,
        },
    }

    result = preprocess_vnwoodknot_for_server(config)

    processed_records = result["processed_records"]
    assert len(processed_records) == 4
    assert all(record["split"] == "train" for record in processed_records)
    assert all("tile_origin_xy" in record for record in processed_records)
    assert all(record["width"] == 1024 for record in processed_records)
    assert all(record["height"] == 1024 for record in processed_records)
    assert all(record["source_image_id"].endswith("train/1/sample") for record in processed_records)
    assert all(len(record["annotations"]) == 1 for record in processed_records)

    manifest_path = processed_root / "manifest.jsonl"
    assert manifest_path.exists()
    manifest_records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert len(manifest_records) == 4
