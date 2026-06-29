"""CSV report writer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from trt_profiler.core.types import ReportData, Reporter


class CsvReporter(Reporter):
    """Write report tables as CSV files.

    Config Keys
    -----------
    output_dir : str, optional
        Directory where CSV files are written. Defaults to ``"csv_report"``.
    tables : list[str], optional
        Table names to export. Defaults to all tables in ``ReportData``.
    """

    def write(self, report_data: ReportData) -> None:
        """Write report tables as CSV files.

        Parameters
        ----------
        report_data
            Report-ready data.
        """

        output_dir = Path(str(self.config.get("output_dir", "csv_report")))
        output_dir.mkdir(parents=True, exist_ok=True)
        configured_tables = self.config.get("tables")
        table_names = (
            [str(item) for item in configured_tables]
            if isinstance(configured_tables, list)
            else list(report_data.tables)
        )
        for table_name in table_names:
            rows = report_data.tables.get(table_name, [])
            _write_rows(output_dir / f"{table_name}.csv", rows)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = _columns(rows)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_value(row.get(column, "")) for column in columns})


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns or ["empty"]


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return str(value)
    return value
