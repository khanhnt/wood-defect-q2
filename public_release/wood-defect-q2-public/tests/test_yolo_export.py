from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.datasets.yolo_export import export_manifest_to_yolo


def _make_annotation(class_name: str, class_id: int, box: list[float]) -> dict:
    x1, y1, x2, y2 = box
    return {
        "class_name": class_name,
        "class_id": class_id,
        "bbox_xyxy_norm": box,
        "bbox_width_norm": x2 - x1,
        "bbox_height_norm": y2 - y1,
        "bbox_area_norm": (x2 - x1) * (y2 - y1),
        "is_small_defect": False,
        "source_label": class_name,
    }


def test_export_manifest_to_yolo(tmp_path):
    image_root = tmp_path / "images_root"
    image_path = image_root / "images" / "train" / "tile_a.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1024, 1024), color=(128, 128, 128)).save(image_path)

    manifest_path = tmp_path / "manifest.jsonl"
    record = {
        "dataset_name": "toy",
        "image_id": "images/train/tile_a",
        "image_path": "images/train/tile_a.jpg",
        "source_image_id": "src_a",
        "split": "train",
        "width": 1024,
        "height": 1024,
        "annotations": [_make_annotation("resin", 0, [0.25, 0.25, 0.5, 0.5])],
        "is_empty": False,
        "empty_reason": None,
        "num_small_annotations": 0,
    }
    manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = export_manifest_to_yolo(
        input_manifest_path=manifest_path,
        image_root_dir=image_root,
        output_root_dir=tmp_path / "yolo_export",
        dataset_name="toy_yolo",
        classes=["resin"],
        prefer_symlink=False,
    )

    dataset_yaml = Path(result["dataset_yaml_path"])
    assert dataset_yaml.exists()
    label_files = list((tmp_path / "yolo_export" / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 1
    assert label_files[0].read_text(encoding="utf-8").strip().startswith("0 ")
