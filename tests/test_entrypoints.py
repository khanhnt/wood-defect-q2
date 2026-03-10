"""Regression checks for script-level model builder calls."""

from pathlib import Path


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_train_script_uses_model_config_keyword():
    content = _read_file("scripts/train.py")
    assert "build_baseline_detector(model_config=model_cfg" in content
    assert "build_baseline_detector(model_cfg=model_cfg" not in content


def test_evaluate_script_uses_model_config_keyword():
    content = _read_file("scripts/evaluate.py")
    assert "build_baseline_detector(model_config=model_cfg" in content
    assert "build_baseline_detector(model_cfg=model_cfg" not in content
