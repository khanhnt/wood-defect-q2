"""Unit tests for preprocessing helpers."""

from src.datasets.base_dataset import build_annotation
from src.datasets.server_preprocessing import generate_tile_starts, remap_annotations_to_tile


def test_generate_tile_starts_covers_axis():
    starts = generate_tile_starts(length=2800, tile_size=1024, overlap=128)
    assert starts == [0, 896, 1776]


def test_remap_annotations_to_tile_keeps_visible_box():
    annotation = build_annotation(
        class_name="live_knot",
        bbox_xyxy_norm=[0.1, 0.25, 0.3, 0.75],
    )
    remapped = remap_annotations_to_tile(
        annotations=[annotation],
        image_width=1000,
        image_height=1000,
        tile_window={"left": 0, "top": 200, "right": 400, "bottom": 800, "width": 400, "height": 600},
        min_visibility=0.5,
    )

    assert len(remapped) == 1
    assert remapped[0]["class_name"] == "live_knot"
    assert remapped[0]["bbox_xyxy_norm"] == [0.25, 0.083333, 0.75, 0.916667]


def test_remap_annotations_to_tile_drops_low_visibility_box():
    annotation = build_annotation(
        class_name="dead_knot",
        bbox_xyxy_norm=[0.75, 0.1, 0.95, 0.3],
    )
    remapped = remap_annotations_to_tile(
        annotations=[annotation],
        image_width=1000,
        image_height=1000,
        tile_window={"left": 0, "top": 0, "right": 800, "bottom": 800, "width": 800, "height": 800},
        min_visibility=0.5,
    )

    assert remapped == []
