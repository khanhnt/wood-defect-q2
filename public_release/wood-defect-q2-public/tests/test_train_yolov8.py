from __future__ import annotations

from scripts.train_yolov8 import resolve_model_source


def test_resolve_model_source_adds_pt_suffix_for_named_model_alias():
    assert resolve_model_source("yolov8m", None) == "yolov8m.pt"


def test_resolve_model_source_keeps_explicit_checkpoint():
    assert resolve_model_source("yolov8s", "custom/best.pt") == "custom/best.pt"


def test_resolve_model_source_keeps_yaml_model_path():
    assert resolve_model_source("configs/models/custom.yaml", None) == "configs/models/custom.yaml"
