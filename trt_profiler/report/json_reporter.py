"""JSON report writer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from trt_profiler.core.types import ReportData, Reporter


class JsonReporter(Reporter):
    """Write report data as JSON.

    Config Keys
    -----------
    path : str, optional
        Output JSON path. Defaults to ``"report.json"``.
    """

    def write(self, report_data: ReportData) -> None:
        """Write a JSON report.

        Parameters
        ----------
        report_data
            Report-ready data.
        """

        output_path = Path(str(self.config.get("path", "report.json")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(report_data), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
