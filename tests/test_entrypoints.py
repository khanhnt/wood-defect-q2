"""Regression checks for script-level model builder calls."""

from pathlib import Path


def _read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_train_script_uses_shared_model_builder():
    content = _read_file("scripts/train.py")
    assert "from src.models.builder import build_model" in content
    assert "model = build_model(model_config=model_cfg, train_config=config.get(\"train\", {}))" in content
    assert "--variant" in content
    assert "--backbone-scale" in content
    assert "--epochs" in content
    assert "--max-train-samples" in content
    assert "--max-val-samples" in content
    assert "_apply_train_overrides" in content


def test_evaluate_script_uses_shared_model_builder():
    content = _read_file("scripts/evaluate.py")
    assert "from src.models.builder import build_model" in content
    assert "model = build_model(model_config=model_cfg, train_config=config.get(\"train\", {}))" in content
    assert "--variant" in content
    assert "--backbone-scale" in content
    assert "--max-samples" in content
    assert "--in-domain-summary-path" in content
    assert "--score-threshold" in content
    assert "_apply_eval_overrides" in content
