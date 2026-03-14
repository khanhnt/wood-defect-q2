"""Unit tests for source-level occurrence and co-occurrence statistics."""

import json

from src.datasets.occurrence_stats import (
    aggregate_source_level_records,
    build_cooccurrence_statistics,
    build_occurrence_statistics,
    compare_manifest_occurrence_statistics,
)


def _record(image_id, annotations, source_image_id=None, split=None):
    return {
        "image_id": image_id,
        "source_image_id": source_image_id,
        "split": split,
        "annotations": annotations,
    }


def _ann(class_name):
    return {"class_name": class_name}


def test_aggregate_source_level_records_groups_tiles_by_source_image_id():
    records = [
        _record("tile_a0", [_ann("live_knot")], source_image_id="img_a", split="train"),
        _record("tile_a1", [_ann("dead_knot")], source_image_id="img_a", split="train"),
        _record("tile_b0", [_ann("crack")], source_image_id="img_b", split="val"),
    ]

    grouped, class_names = aggregate_source_level_records(records)

    assert class_names == ["crack", "dead_knot", "live_knot"]
    assert len(grouped) == 2
    assert grouped[0]["source_image_id"] == "img_a"
    assert len(grouped[0]["annotations"]) == 2
    assert grouped[0]["tile_count"] == 2


def test_build_occurrence_statistics_counts_singletons_and_single_class_images():
    records = [
        _record("img_1", [_ann("live_knot")]),
        _record("img_2", [_ann("live_knot"), _ann("live_knot")]),
        _record("img_3", [_ann("dead_knot"), _ann("crack")]),
        _record("img_4", []),
    ]

    per_class_df, summary = build_occurrence_statistics(records)
    rows = {row["class_name"]: row for row in per_class_df.to_dict("records")}

    assert summary["num_images"] == 4
    assert summary["num_empty_images"] == 1
    assert summary["num_single_annotation_images"] == 1
    assert summary["num_single_class_images"] == 2
    assert summary["num_multi_class_images"] == 1

    assert rows["live_knot"]["images_with_class"] == 2
    assert rows["live_knot"]["images_only_this_class"] == 2
    assert rows["live_knot"]["images_exactly_one_annotation_total"] == 1
    assert rows["live_knot"]["images_only_this_class_multi_annotation"] == 1

    assert rows["dead_knot"]["images_multi_class_with_class"] == 1
    assert rows["crack"]["images_multi_class_with_class"] == 1


def test_build_cooccurrence_statistics_counts_pairs_and_triples():
    records = [
        _record("img_1", [_ann("live_knot"), _ann("dead_knot"), _ann("crack")]),
        _record("img_2", [_ann("live_knot"), _ann("dead_knot")]),
        _record("img_3", [_ann("dead_knot"), _ann("crack")]),
    ]

    pair_df, triple_df, summary = build_cooccurrence_statistics(records)

    pair_rows = {row["combo_key"]: row for row in pair_df.to_dict("records")}
    triple_rows = {row["combo_key"]: row for row in triple_df.to_dict("records")}

    assert summary["num_positive_source_images"] == 3
    assert pair_rows["dead_knot + live_knot"]["source_image_count"] == 2
    assert pair_rows["crack + dead_knot"]["source_image_count"] == 2
    assert pair_rows["crack + live_knot"]["source_image_count"] == 1
    assert triple_rows["crack + dead_knot + live_knot"]["source_image_count"] == 1


def test_compare_manifest_occurrence_statistics_combines_multiple_manifests(tmp_path):
    benchmark_a = tmp_path / "benchmark_a.jsonl"
    benchmark_b = tmp_path / "benchmark_b.jsonl"
    records_a = [
        _record("tile_a0", [_ann("live_knot"), _ann("dead_knot")], source_image_id="img_a", split="train"),
        _record("tile_a1", [_ann("live_knot")], source_image_id="img_a", split="train"),
        _record("tile_b0", [_ann("resin")], source_image_id="img_b", split="val"),
    ]
    records_b = [
        _record("tile_c0", [_ann("crack"), _ann("dead_knot")], source_image_id="img_c", split="train"),
        _record("tile_d0", [_ann("marrow")], source_image_id="img_d", split="val"),
    ]
    benchmark_a.write_text("\n".join(json.dumps(record) for record in records_a) + "\n", encoding="utf-8")
    benchmark_b.write_text("\n".join(json.dumps(record) for record in records_b) + "\n", encoding="utf-8")

    overview_df, per_class_df, pair_df, triple_df, summary = compare_manifest_occurrence_statistics(
        manifest_specs=[
            {"benchmark_name": "random", "selection_mode": "random", "manifest_path": benchmark_a},
            {"benchmark_name": "stratified", "selection_mode": "stratified", "manifest_path": benchmark_b},
        ],
        class_names=["live_knot", "dead_knot", "resin", "crack", "marrow"],
        splits=["all", "train", "val"],
        top_k=5,
    )

    assert set(overview_df["benchmark_name"]) == {"random", "stratified"}
    assert set(overview_df["split"]) == {"all", "train", "val"}
    assert set(per_class_df["selection_mode"]) == {"random", "stratified"}
    assert "dead_knot + live_knot" in set(pair_df["combo_key"])
    assert triple_df.empty
    assert summary["splits"] == ["all", "train", "val"]
