"""Tests for environment-variable based path config expansion."""

from pathlib import Path

import pytest

from src.utils.config import expand_path


def test_expand_path_resolves_environment_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOD_TEST_PATH", str(tmp_path))
    resolved = expand_path("${WOOD_TEST_PATH}/manifest.jsonl")
    assert resolved == Path(tmp_path) / "manifest.jsonl"


def test_expand_path_raises_for_unresolved_environment_variable(monkeypatch):
    monkeypatch.delenv("WOOD_MISSING_PATH", raising=False)

    with pytest.raises(ValueError, match="WOOD_MISSING_PATH"):
        expand_path("${WOOD_MISSING_PATH}/manifest.jsonl")
