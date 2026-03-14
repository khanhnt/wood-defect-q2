"""Checks for screened benchmark manifest generation."""

from src.datasets.screened_benchmark import (
    build_screened_processed_records,
    select_screened_source_ids,
)


def _record(source_image_id, split, image_id, annotations):
    return {
        "dataset_name": "toy",
        "image_id": image_id,
        "image_path": f"images/{image_id}.jpg",
        "split": split,
        "source_image_id": source_image_id,
        "width": 1024,
        "height": 1024,
        "annotations": annotations,
        "is_empty": len(annotations) == 0,
        "empty_reason": None if annotations else "negative_tile",
        "num_small_annotations": sum(1 for ann in annotations if ann.get("is_small_defect")),
    }


def _ann(class_name, class_id, is_small=False):
    return {
        "class_name": class_name,
        "class_id": class_id,
        "bbox_xyxy_norm": [0.1, 0.1, 0.2, 0.2],
        "bbox_width_norm": 0.1,
        "bbox_height_norm": 0.1,
        "bbox_area_norm": 0.01,
        "is_small_defect": is_small,
    }


def test_select_screened_source_ids_respects_split_targets():
    records = [
        _record("train/a", "train", "tile_a_0", [_ann("live_knot", 0)]),
        _record("train/b", "train", "tile_b_0", [_ann("dead_knot", 1)]),
        _record("train/c", "train", "tile_c_0", [_ann("resin", 2)]),
        _record("val/a", "val", "tile_d_0", [_ann("live_knot", 0)]),
        _record("test/a", "test", "tile_e_0", [_ann("crack", 4)]),
    ]

    selected_source_ids, summary = select_screened_source_ids(
        processed_records=records,
        kept_classes=["live_knot", "dead_knot", "resin", "crack"],
        target_source_images=5,
        seed=42,
    )

    assert len(selected_source_ids) == 5
    assert summary["selected_source_images_by_split"]["train"] == 3
    assert summary["selected_source_images_by_split"]["val"] == 1
    assert summary["selected_source_images_by_split"]["test"] == 1


def test_build_screened_processed_records_remaps_class_ids_and_keeps_negative_tiles():
    records = [
        _record("train/a", "train", "tile_a_0", [_ann("live_knot", 5, is_small=True)]),
        _record("train/a", "train", "tile_a_1", []),
        _record("train/b", "train", "tile_b_0", [_ann("blue_stain", 8)]),
    ]

    screened_records = build_screened_processed_records(
        processed_records=records,
        selected_source_ids={"train/a"},
        kept_classes=["live_knot", "dead_knot"],
        dataset_name="screened",
    )

    assert len(screened_records) == 2
    positive_record = screened_records[0]
    negative_record = screened_records[1]
    assert positive_record["dataset_name"] == "screened"
    assert positive_record["annotations"][0]["class_name"] == "live_knot"
    assert positive_record["annotations"][0]["class_id"] == 0
    assert positive_record["num_small_annotations"] == 1
    assert negative_record["annotations"] == []
    assert negative_record["is_empty"] is True


def test_select_screened_source_ids_supports_multiple_selection_modes():
    records = [
        _record("train/a", "train", "tile_a_0", [_ann("live_knot", 0)]),
        _record("train/b", "train", "tile_b_0", [_ann("dead_knot", 1)]),
        _record("train/c", "train", "tile_c_0", [_ann("resin", 2)]),
        _record("train/d", "train", "tile_d_0", [_ann("crack", 4)]),
        _record("train/e", "train", "tile_e_0", [_ann("knot_missing", 6)]),
        _record("val/a", "val", "tile_f_0", [_ann("live_knot", 0), _ann("crack", 4)]),
        _record("val/b", "val", "tile_g_0", [_ann("marrow", 5)]),
        _record("test/a", "test", "tile_h_0", [_ann("dead_knot", 1), _ann("resin", 2)]),
    ]

    for selection_mode in ("random", "stratified", "rare_first"):
        selected_source_ids, summary = select_screened_source_ids(
            processed_records=records,
            kept_classes=["live_knot", "dead_knot", "resin", "crack", "marrow", "knot_missing"],
            target_source_images=6,
            seed=42,
            selection_mode=selection_mode,
        )

        assert len(selected_source_ids) == 6
        assert summary["selection_mode"] == selection_mode
        assert "selected_source_presence_by_split" in summary
