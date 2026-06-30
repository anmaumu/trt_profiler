from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from trt_profiler.report.dash_app import (
    build_heatmap_figure,
    build_matrix_figure,
    build_overlay_figure,
    build_ranking_figure,
    load_dashboard_rows,
    metric_help_for,
)


def test_load_dashboard_rows_supports_multiple_reports(tmp_path: Path) -> None:
    report_a = _write_report(tmp_path / "run_a" / "report.json", "a_vs_b", 0.1)
    report_b = _write_report(tmp_path / "run_b" / "report.json", "a_vs_c", 0.2)

    rows = load_dashboard_rows([report_a, report_b])

    assert {row["report"] for row in rows["metric_summary"]} == {"run_a", "run_b"}
    assert {row["comparison"] for row in rows["metric_summary"]} == {"a_vs_b", "a_vs_c"}


def test_dash_figures_build_from_rows(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "run" / "report.json", "a_vs_b", 0.1)
    rows = load_dashboard_rows([report])

    matrix = build_matrix_figure(rows["metric_summary"])
    ranking = build_ranking_figure(rows["metric_summary"])

    assert matrix.data
    assert ranking.data


def test_heatmap_and_overlay_preview_figures(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    heatmap_path = tmp_path / "heatmap.npy"
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(image_path)
    np.save(heatmap_path, np.ones((4, 4), dtype=np.float32))

    heatmap = build_heatmap_figure(heatmap_path)
    overlay = build_overlay_figure(image_path, heatmap_path)

    assert heatmap.data
    assert len(overlay.data) == 2


def test_metric_help_resolves_registered_metric() -> None:
    help_data = metric_help_for("classification_consistency")

    assert help_data["title"] == "Classification Metrics"
    assert "top1_match_rate" in help_data["stats"]


def _write_report(path: Path, comparison: str, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"model": {"name": "dummy"}, "variants": ["a", "b"]},
        "summary": {},
        "tables": {
            "metric_summary": [
                {
                    "comparison": comparison,
                    "stage": "raw",
                    "metric": "tensor_diff",
                    "output": "logits",
                    "stat": "mean_abs_error",
                    "value": value,
                }
            ],
            "per_sample": [
                {
                    "sample_id": "sample_1",
                    "comparison": comparison,
                    "stage": "raw",
                    "metric": "tensor_diff",
                    "output": "logits",
                    "stat": "mean_abs_error",
                    "value": value,
                }
            ],
            "failed_cases": [],
        },
        "artifacts": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
