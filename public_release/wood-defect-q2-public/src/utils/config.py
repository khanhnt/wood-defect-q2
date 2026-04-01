"""Config utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_path(value: str | Path | None) -> Path | None:
    """Expand user and environment variables for a path-like config value."""
    if value in {None, ""}:
        return None

    raw_value = str(value)
    expanded_value = os.path.expandvars(raw_value)
    unresolved_variables = sorted(
        {
            match.group(1) or match.group(2)
            for match in re.finditer(r"\$([A-Za-z_][A-Za-z0-9_]*)|\$\{([^}]+)\}", expanded_value)
        }
    )
    if unresolved_variables:
        variable_list = ", ".join(unresolved_variables)
        raise ValueError(
            "Unresolved environment variable(s) in path config: "
            f"{variable_list}. Export them before running. Raw value: {raw_value}"
        )

    return Path(expanded_value).expanduser()
