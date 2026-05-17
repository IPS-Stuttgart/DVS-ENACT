"""Shared report-writing utilities for EventVOT report scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write *rows* to a CSV file at *path*, creating parent dirs as needed.

    An empty file is created when *rows* is empty.  Fieldnames are collected
    from all rows in insertion order, which handles heterogeneous dicts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
