"""JSON report loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trt_profiler.core.types import ReportData


def load_report_data(path: str | Path) -> ReportData:
    """Load report data from a JSON report.

    Parameters
    ----------
    path
        JSON report path.

    Returns
    -------
    ReportData
        Parsed report data.

    Raises
    ------
    ValueError
        If the JSON root is not an object.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Report JSON must contain an object.")
    return ReportData(
        metadata=_dict(raw.get("metadata")),
        summary=_dict(raw.get("summary")),
        tables=_tables(raw.get("tables")),
        artifacts=_dict(raw.get("artifacts")),
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _tables(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for key, rows in value.items():
        if isinstance(rows, list):
            tables[str(key)] = [row for row in rows if isinstance(row, dict)]
    return tables
