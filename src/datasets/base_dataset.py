"""Base dataset utilities for wood defect detection."""

from typing import Any, Dict


class BaseWoodDefectDataset:
    """Base class for dataset implementations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError
