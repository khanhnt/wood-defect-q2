import numpy as np

from src.datasets.label_mapping import (
    remap_predictions_and_targets_for_cross_dataset,
    resolve_cross_dataset_label_mapping,
)


def test_resolve_cross_dataset_label_mapping_reports_mapped_ignored_and_unmatched():
    report = resolve_cross_dataset_label_mapping(
        source_class_names=[
            "live_knot",
            "dead_knot",
            "resin",
            "crack",
        ],
        target_class_names=[
            "live_knot",
            "dead_knot",
            "knot_free",
        ],
        label_mapping={
            "live_knot": "live_knot",
            "dead_knot": "dead_knot",
            "knot_free": None,
        },
    )

    mapping_table = report["mapping_table"]
    assert report["mapped_class_names"] == ["live_knot", "dead_knot"]
    assert report["mapped_target_classes"] == ["live_knot", "dead_knot"]
    assert report["ignored_target_classes"] == ["knot_free"]
    assert report["unmatched_target_classes"] == []
    assert report["unmatched_source_classes"] == ["resin", "crack"]

    status_by_target = {
        row["target_class_name"]: row["status"]
        for row in mapping_table.to_dict("records")
        if row["target_class_name"] is not None
    }
    assert status_by_target["live_knot"] == "mapped"
    assert status_by_target["dead_knot"] == "mapped"
    assert status_by_target["knot_free"] == "ignored_target_class"

    unmatched_source_rows = mapping_table.loc[
        mapping_table["status"] == "unmatched_source_class", "source_class_name"
    ].tolist()
    assert unmatched_source_rows == ["resin", "crack"]


def test_remap_predictions_and_targets_for_cross_dataset_filters_to_overlap_only():
    report = resolve_cross_dataset_label_mapping(
        source_class_names=["live_knot", "dead_knot", "resin"],
        target_class_names=["live_knot", "dead_knot", "knot_free"],
        label_mapping={
            "live_knot": "live_knot",
            "dead_knot": "dead_knot",
            "knot_free": None,
        },
    )

    predictions = [
        {
            "image_id": "image-1",
            "boxes": np.asarray(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [1.0, 1.0, 8.0, 8.0],
                    [2.0, 2.0, 9.0, 9.0],
                ],
                dtype=np.float32,
            ),
            "labels": np.asarray([0, 1, 2], dtype=np.int64),
            "scores": np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        }
    ]
    targets = [
        {
            "image_id": "image-1",
            "boxes": np.asarray(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [1.0, 1.0, 8.0, 8.0],
                ],
                dtype=np.float32,
            ),
            "labels": np.asarray([0, 2], dtype=np.int64),
        }
    ]

    remapped_predictions, remapped_targets, remap_counts = remap_predictions_and_targets_for_cross_dataset(
        predictions=predictions,
        targets=targets,
        mapping_report=report,
    )

    assert remapped_predictions[0]["labels"].tolist() == [0, 1]
    assert remapped_targets[0]["labels"].tolist() == [0]
    assert remap_counts == {
        "ignored_prediction_count": 1,
        "ignored_target_annotation_count": 1,
        "mapped_prediction_count": 2,
        "mapped_target_annotation_count": 1,
    }
