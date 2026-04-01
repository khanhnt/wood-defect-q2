"""Basic dataset tests."""


def test_placeholder_dataset_import() -> None:
    from src.datasets.base_dataset import BaseWoodDefectDataset
    assert BaseWoodDefectDataset is not None
