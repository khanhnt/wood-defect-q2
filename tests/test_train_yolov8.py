from __future__ import annotations

from pathlib import Path

from scripts.train_yolov8 import _resolve_model_reference


def test_resolve_model_reference_keeps_builtin_name():
    assert _resolve_model_reference("yolov8s.pt") == "yolov8s.pt"


def test_resolve_model_reference_resolves_existing_path(tmp_path):
    model_path = tmp_path / "yolov8s-p2.yaml"
    model_path.write_text("nc: 7\n", encoding="utf-8")
    assert _resolve_model_reference(str(model_path)) == str(model_path.resolve())


def test_resolve_model_reference_handles_blank_values():
    assert _resolve_model_reference("") is None
    assert _resolve_model_reference("   ") is None
    assert _resolve_model_reference(None) is None
