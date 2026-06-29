from __future__ import annotations

from pathlib import Path

from trt_profiler.core.types import ReportData
from trt_profiler.report.csv_reporter import CsvReporter


def test_csv_reporter_writes_report_tables(tmp_path: Path) -> None:
    report_data = ReportData(
        metadata={},
        summary={},
        tables={
            "metric_summary": [
                {
                    "comparison": "ort_cpu_vs_trt",
                    "stage": "raw",
                    "metric": "tensor_diff",
                    "value": 0.123456789,
                }
            ],
            "per_sample": [],
        },
        artifacts={},
    )

    CsvReporter(config={"output_dir": str(tmp_path)}).write(report_data)

    summary_csv = tmp_path / "metric_summary.csv"
    sample_csv = tmp_path / "per_sample.csv"
    assert summary_csv.exists()
    assert sample_csv.exists()
    assert "ort_cpu_vs_trt" in summary_csv.read_text(encoding="utf-8")
