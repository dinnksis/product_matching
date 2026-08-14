"""Shared, dependency-light helpers for the frozen validation protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def validation_split_paths(
    prepared_dir: Path,
    specs: Sequence[str],
) -> dict[str, Path]:
    values = specs or ["iid=val_pairs.parquet"]
    result: dict[str, Path] = {}
    for spec in values:
        name, separator, raw_path = spec.partition("=")
        name = name.strip().lower()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(
                f"Invalid --validation-split {spec!r}; expected NAME=PATH"
            )
        if not name.replace("_", "").isalnum():
            raise ValueError(
                f"Validation split name must contain only letters, digits and underscores: {name!r}"
            )
        if name in result:
            raise ValueError(f"Duplicate validation split name: {name!r}")
        path = Path(raw_path)
        result[name] = path if path.is_absolute() else prepared_dir / path
    return result
