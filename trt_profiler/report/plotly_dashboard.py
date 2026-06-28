"""Plotly HTML dashboard reporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trt_profiler.core.types import ReportData, Reporter


class PlotlyDashboardReporter(Reporter):
    """Write an interactive Plotly HTML dashboard from ReportData."""

    def write(self, report_data: ReportData) -> None:
        """Write a Plotly dashboard HTML file.

        Parameters
        ----------
        report_data
            Report-ready data.

        Raises
        ------
        RuntimeError
            If Plotly is not installed.
        """

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as exc:
            raise RuntimeError("plotly is required for PlotlyDashboardReporter.") from exc

        output_path = Path(str(self.config.get("path", "dashboard.html")))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary_rows = report_data.tables.get("metric_summary", [])
        sample_rows = report_data.tables.get("per_sample", [])
        failed_rows = report_data.tables.get("failed_cases", [])

        fig = make_subplots(
            rows=3,
            cols=1,
            specs=[[{"type": "table"}], [{"type": "bar"}], [{"type": "table"}]],
            row_heights=[0.42, 0.28, 0.30],
            vertical_spacing=0.08,
            subplot_titles=(
                "Metric Summary",
                "Sample Count by Metric",
                "Failed Cases",
            ),
        )

        fig.add_trace(_table_trace(summary_rows, _summary_columns()), row=1, col=1)
        fig.add_trace(_sample_count_trace(sample_rows, go), row=2, col=1)
        fig.add_trace(_table_trace(failed_rows, _failed_columns()), row=3, col=1)

        title = _dashboard_title(report_data.metadata)
        fig.update_layout(
            title=title,
            height=int(self.config.get("height", 1100)),
            margin={"l": 24, "r": 24, "t": 90, "b": 24},
            showlegend=False,
        )
        output_path.write_text(
            fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8"
        )


def _dashboard_title(metadata: dict[str, Any]) -> str:
    model = metadata.get("model", {})
    model_name = model.get("name", "model") if isinstance(model, dict) else "model"
    variants = metadata.get("variants", [])
    variant_text = ", ".join(str(item) for item in variants) if isinstance(variants, list) else ""
    return f"TRT Profiler Dashboard - {model_name}<br><sup>{variant_text}</sup>"


def _summary_columns() -> list[str]:
    return ["comparison", "stage", "metric", "output", "stat", "value"]


def _failed_columns() -> list[str]:
    return ["sample_id", "comparison", "stage", "metric", "output", "stat", "value", "status"]


def _table_trace(rows: list[dict[str, Any]], columns: list[str]) -> Any:
    import plotly.graph_objects as go

    display_rows = rows[:200]
    values = [[_format_cell(row.get(column, "")) for row in display_rows] for column in columns]
    return go.Table(
        header={
            "values": columns,
            "fill_color": "#1f2937",
            "font": {"color": "white", "size": 12},
            "align": "left",
        },
        cells={
            "values": values,
            "fill_color": "#f8fafc",
            "font": {"color": "#111827", "size": 11},
            "align": "left",
            "height": 24,
        },
    )


def _sample_count_trace(rows: list[dict[str, Any]], go: Any) -> Any:
    counts: dict[str, int] = {}
    for row in rows:
        metric = str(row.get("metric", "unknown"))
        counts[metric] = counts.get(metric, 0) + 1

    if not counts:
        counts = {"no_sample_rows": 0}

    return go.Bar(
        x=list(counts),
        y=list(counts.values()),
        marker={"color": "#2563eb"},
    )


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)
