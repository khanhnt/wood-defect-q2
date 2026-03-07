"""Basic evaluation pipeline import test."""


def test_evaluator_import() -> None:
    from src.engine.evaluator import Evaluator
    assert Evaluator is not None
