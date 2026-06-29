from __future__ import annotations

from pathlib import Path

from trt_profiler.core.types import ReportData
from trt_profiler.report.plotly_dashboard import PlotlyDashboardReporter


def test_plotly_dashboard_writes_interactive_html(tmp_path: Path) -> None:
    output_path = tmp_path / "dashboard.html"
    report_data = ReportData(
        metadata={
            "model": {"name": "dummy"},
            "variants": ["ort_cpu", "trt_fp16"],
            "comparisons": ["ort_cpu_vs_trt_fp16"],
        },
        summary={},
        tables={
            "metric_summary": [
                {
                    "comparison": "ort_cpu_vs_trt_fp16",
                    "stage": "raw",
                    "metric": "tensor_diff",
                    "output": "logits",
                    "stat": "mean_abs_error",
                    "value": 0.01,
                }
            ],
            "per_sample": [
                {
                    "sample_id": "sample_1",
                    "comparison": "ort_cpu_vs_trt_fp16",
                    "stage": "raw",
                    "metric": "tensor_diff",
                    "output": "logits",
                    "stat": "mean_abs_error",
                    "value": 0.01,
                    "status": None,
                }
            ],
            "failed_cases": [],
        },
        artifacts={},
    )

    PlotlyDashboardReporter(config={"path": str(output_path)}).write(report_data)

    html = output_path.read_text(encoding="utf-8")
    assert "comparisonSelect" in html
    assert "comparisonMatrix" in html
    assert "metricRanking" in html
    assert "Plotly.react" in html
    assert "ort_cpu_vs_trt_fp16" in html
