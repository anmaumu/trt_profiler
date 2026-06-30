"""Dash server dashboard for interactive report exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from trt_profiler.report.loader import load_report_data


def run_dash_server(report_paths: list[str | Path], host: str, port: int) -> None:
    """Run the Dash report viewer.

    Parameters
    ----------
    report_paths
        One or more report JSON paths.
    host
        Host interface for the Dash server.
    port
        Port for the Dash server.

    Raises
    ------
    RuntimeError
        If Dash is not installed.
    """

    try:
        from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html
    except ImportError as exc:
        raise RuntimeError("dash is required. Install trt-profiler[dashboard].") from exc

    dash_table_component: Any = dash_table
    rows = load_dashboard_rows(report_paths)
    app = Dash(__name__)
    app.title = "TRT Profiler Dash"
    app.layout = html.Div(
        [
            html.Header(
                [
                    html.H1("TRT Profiler Dash"),
                    html.Div("複数report横断 / comparison matrix / preview viewer"),
                ],
                style=_header_style(),
            ),
            html.Main(
                [
                    html.Div(
                        [
                            _dropdown(dcc, "report-filter", "Report"),
                            _dropdown(dcc, "comparison-filter", "比較ペア"),
                            _dropdown(dcc, "stage-filter", "Stage"),
                            _dropdown(dcc, "metric-filter", "Metric"),
                            html.Button(
                                "?", id="metric-help-open", n_clicks=0, style=_help_button_style()
                            ),
                        ],
                        style=_controls_style(),
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2("Metric Help", style={"margin": "0"}),
                                            html.Button(
                                                "Close",
                                                id="metric-help-close",
                                                n_clicks=0,
                                                style=_close_button_style(),
                                            ),
                                        ],
                                        style=_modal_header_style(),
                                    ),
                                    html.Div(id="metric-help-body"),
                                ],
                                style=_modal_content_style(),
                            )
                        ],
                        id="metric-help-modal",
                        style=_modal_overlay_style(hidden=True),
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="comparison-matrix"),
                            dcc.Graph(id="metric-ranking"),
                        ],
                        style=_grid_style(),
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="sample-distribution"),
                            dcc.Graph(id="summary-bar"),
                        ],
                        style=_grid_style(),
                    ),
                    html.H2("Summary"),
                    dash_table_component.DataTable(
                        id="summary-table",
                        page_size=12,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                    ),
                    html.H2("Failed / Selected Cases"),
                    dash_table_component.DataTable(
                        id="failed-table",
                        page_size=8,
                        sort_action="native",
                        filter_action="native",
                        row_selectable="single",
                        style_table={"overflowX": "auto"},
                    ),
                    html.Div(
                        [
                            html.Pre(id="selected-detail", style=_detail_style()),
                            dcc.Graph(id="source-preview"),
                            dcc.Graph(id="heatmap-preview"),
                            dcc.Graph(id="overlay-preview"),
                        ],
                        style=_preview_grid_style(),
                    ),
                ],
                style={"padding": "20px 28px"},
            ),
        ],
        style={"fontFamily": "Inter, system-ui, sans-serif", "backgroundColor": "#f6f8fb"},
    )

    @app.callback(
        Output("report-filter", "options"),
        Output("report-filter", "value"),
        Output("comparison-filter", "options"),
        Output("comparison-filter", "value"),
        Output("stage-filter", "options"),
        Output("stage-filter", "value"),
        Output("metric-filter", "options"),
        Output("metric-filter", "value"),
        Input("report-filter", "value"),
        Input("comparison-filter", "value"),
        Input("stage-filter", "value"),
        Input("metric-filter", "value"),
    )
    def update_filter_options(
        report_value: str | None,
        comparison_value: str | None,
        stage_value: str | None,
        metric_value: str | None,
    ) -> tuple[
        list[dict[str, str]],
        str,
        list[dict[str, str]],
        str,
        list[dict[str, str]],
        str,
        list[dict[str, str]],
        str,
    ]:
        report_values = ["all", *_unique(row["report"] for row in rows["all"])]
        selected_report = _select_value(report_value, report_values)
        report_rows = _filter_report(rows["all"], selected_report)

        comparison_values = _unique(row["comparison"] for row in report_rows)
        selected_comparison = _select_value(comparison_value, comparison_values)
        comparison_rows = [
            row for row in report_rows if row.get("comparison") == selected_comparison
        ]

        stage_values = _unique(row["stage"] for row in comparison_rows)
        selected_stage = _select_value(stage_value, stage_values)
        stage_rows = [row for row in comparison_rows if row.get("stage") == selected_stage]

        metric_values = _unique(row["metric"] for row in stage_rows)
        selected_metric = _select_value(metric_value, metric_values)
        return (
            _options(report_values),
            selected_report,
            _options(comparison_values),
            selected_comparison,
            _options(stage_values),
            selected_stage,
            _options(metric_values),
            selected_metric,
        )

    @app.callback(
        Output("comparison-matrix", "figure"),
        Output("metric-ranking", "figure"),
        Output("sample-distribution", "figure"),
        Output("summary-bar", "figure"),
        Output("summary-table", "columns"),
        Output("summary-table", "data"),
        Output("failed-table", "columns"),
        Output("failed-table", "data"),
        Input("report-filter", "value"),
        Input("comparison-filter", "value"),
        Input("stage-filter", "value"),
        Input("metric-filter", "value"),
    )
    def update_views(
        report_value: str,
        comparison_value: str,
        stage_value: str,
        metric_value: str,
    ) -> tuple[
        go.Figure,
        go.Figure,
        go.Figure,
        go.Figure,
        list[dict[str, str]],
        list[dict[str, Any]],
        list[dict[str, str]],
        list[dict[str, Any]],
    ]:
        report_rows = _filter_report(rows["all"], report_value)
        cross_summary = [
            row
            for row in rows["metric_summary"]
            if _row_in_report(row, report_value)
            and row.get("stage") == stage_value
            and row.get("metric") == metric_value
        ]
        summary_rows = [row for row in cross_summary if row.get("comparison") == comparison_value]
        sample_rows = [
            row
            for row in rows["per_sample"]
            if _row_in_report(row, report_value)
            and row.get("comparison") == comparison_value
            and row.get("stage") == stage_value
            and row.get("metric") == metric_value
        ]
        failed_rows = [
            row
            for row in rows["failed_cases"]
            if _row_in_report(row, report_value)
            and row.get("comparison") == comparison_value
            and row.get("stage") == stage_value
            and row.get("metric") == metric_value
        ]
        if not failed_rows:
            failed_rows = sample_rows[:50]

        del report_rows
        return (
            build_matrix_figure(cross_summary),
            build_ranking_figure(cross_summary),
            build_sample_distribution(sample_rows),
            build_summary_bar(summary_rows),
            _columns(summary_rows),
            summary_rows,
            _columns(failed_rows),
            failed_rows,
        )

    @app.callback(
        Output("selected-detail", "children"),
        Output("source-preview", "figure"),
        Output("heatmap-preview", "figure"),
        Output("overlay-preview", "figure"),
        Input("failed-table", "selected_rows"),
        State("failed-table", "data"),
        State("report-filter", "value"),
        State("comparison-filter", "value"),
        State("stage-filter", "value"),
        State("metric-filter", "value"),
    )
    def update_preview(
        selected_rows: list[int] | None,
        table_rows: list[dict[str, Any]] | None,
        report_value: str,
        comparison_value: str,
        stage_value: str,
        metric_value: str,
    ) -> tuple[str, go.Figure, go.Figure, go.Figure]:
        selected = _selected_row(selected_rows, table_rows)
        if selected is None:
            selected = _first_row_for_preview(
                rows["per_sample"],
                report_value,
                comparison_value,
                stage_value,
                metric_value,
            )
        if selected is None:
            return (
                "No selected row.",
                _blank_figure("source image"),
                _blank_figure("heatmap"),
                _blank_figure("overlay"),
            )

        heatmap_path = _heatmap_path_for_row(selected, rows["per_sample"])
        source_path = _path_from_row(selected, "source_path")
        detail = _detail_text(selected, heatmap_path, source_path)
        return (
            detail,
            build_image_figure(source_path),
            build_heatmap_figure(heatmap_path),
            build_overlay_figure(source_path, heatmap_path),
        )

    @app.callback(
        Output("metric-help-modal", "style"),
        Output("metric-help-body", "children"),
        Input("metric-help-open", "n_clicks"),
        Input("metric-help-close", "n_clicks"),
        Input("metric-filter", "value"),
        State("metric-help-modal", "style"),
    )
    def update_metric_help(
        open_clicks: int | None,
        close_clicks: int | None,
        metric_value: str | None,
        current_style: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], Any]:
        del open_clicks, close_clicks
        help_data = metric_help_for(metric_value or "")
        body = _metric_help_component(html, help_data)
        triggered = ctx.triggered_id
        if triggered == "metric-help-close":
            return _modal_overlay_style(hidden=True), body
        if triggered == "metric-help-open":
            return _modal_overlay_style(hidden=False), body
        if current_style and current_style.get("display") == "flex":
            return _modal_overlay_style(hidden=False), body
        return _modal_overlay_style(hidden=True), body

    app.run(host=host, port=port, debug=False)


def load_dashboard_rows(report_paths: list[str | Path]) -> dict[str, list[dict[str, Any]]]:
    """Load one or more reports and flatten their tables for Dash.

    Parameters
    ----------
    report_paths
        Report JSON paths.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Rows grouped by table name plus an ``all`` table.
    """

    tables: dict[str, list[dict[str, Any]]] = {
        "metric_summary": [],
        "per_sample": [],
        "failed_cases": [],
        "all": [],
    }
    for path_value in report_paths:
        path = Path(path_value)
        report = load_report_data(path)
        report_name = path.stem
        if len(report_paths) > 1:
            report_name = path.parent.name or path.stem
        for table_name in list(tables):
            if table_name == "all":
                continue
            for row in report.tables.get(table_name, []):
                enriched = dict(row)
                enriched["report"] = report_name
                enriched["_report_path"] = str(path)
                enriched["_report_dir"] = str(path.parent)
                tables[table_name].append(enriched)
                tables["all"].append(enriched)
    return tables


def build_matrix_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Build a comparison matrix heatmap."""

    numeric_rows = [row for row in rows if _numeric(row.get("value")) is not None]
    comparisons = _unique(str(row.get("comparison", "")) for row in numeric_rows)
    stats = _unique(_row_stat_label(row) for row in numeric_rows)
    z = [
        [
            _numeric(
                next(
                    (
                        row.get("value")
                        for row in numeric_rows
                        if row.get("comparison") == comparison and _row_stat_label(row) == stat
                    ),
                    None,
                )
            )
            for comparison in comparisons
        ]
        for stat in stats
    ]
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=comparisons,
                y=stats,
                z=z,
                colorscale="Blues",
                hovertemplate="%{x}<br>%{y}<br>%{z:.6g}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title="Comparison matrix", margin={"l": 150, "r": 24, "t": 56, "b": 120})
    fig.update_xaxes(tickangle=-35, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def build_ranking_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Build a metric ranking graph."""

    values: list[tuple[str, float]] = []
    for row in rows:
        value = _numeric(row.get("value"))
        if value is None:
            continue
        values.append((f"{row.get('comparison')} / {_row_stat_label(row)}", value))
    ranked = sorted(values, key=lambda item: abs(item[1]), reverse=True)[:30]
    ranked.reverse()
    fig = go.Figure(
        data=[
            go.Bar(
                x=[value for _, value in ranked],
                y=[label for label, _ in ranked],
                orientation="h",
                marker={"color": "#0f766e"},
            )
        ]
    )
    fig.update_layout(title="Metric ranking", margin={"l": 220, "r": 24, "t": 56, "b": 40})
    return fig


def build_summary_bar(rows: list[dict[str, Any]]) -> go.Figure:
    """Build a selected-comparison summary bar chart."""

    numeric_rows = [row for row in rows if _numeric(row.get("value")) is not None]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[_row_stat_label(row) for row in numeric_rows],
                y=[_numeric(row.get("value")) for row in numeric_rows],
                marker={"color": "#2563eb"},
            )
        ]
    )
    fig.update_layout(
        title="Selected comparison summary", margin={"l": 56, "r": 24, "t": 56, "b": 120}
    )
    fig.update_xaxes(tickangle=-35, automargin=True)
    return fig


def build_sample_distribution(rows: list[dict[str, Any]]) -> go.Figure:
    """Build per-sample distribution chart."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _numeric(row.get("value"))
        if value is None:
            continue
        grouped.setdefault(_row_stat_label(row), []).append(value)
    fig = go.Figure()
    for label, values in grouped.items():
        fig.add_trace(go.Box(name=label, y=values, boxpoints="all", jitter=0.35))
    fig.update_layout(title="Per-sample distribution", margin={"l": 56, "r": 24, "t": 56, "b": 110})
    fig.update_xaxes(tickangle=-35, automargin=True)
    return fig


def build_image_figure(path: Path | None) -> go.Figure:
    """Build source image preview figure."""

    if path is None or not path.exists():
        return _blank_figure("source image")
    image = _load_image_array(path)
    if image is None:
        return _blank_figure("source image")
    fig = go.Figure(go.Image(z=image))
    fig.update_layout(title=f"Source image: {path.name}", margin={"l": 8, "r": 8, "t": 42, "b": 8})
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def build_heatmap_figure(path: Path | None) -> go.Figure:
    """Build heatmap preview figure."""

    if path is None or not path.exists():
        return _blank_figure("heatmap")
    heatmap = _load_heatmap(path)
    if heatmap is None:
        return _blank_figure("heatmap")
    fig = go.Figure(go.Heatmap(z=heatmap, colorscale="Inferno"))
    fig.update_layout(title=f"Heatmap: {path.name}", margin={"l": 8, "r": 8, "t": 42, "b": 8})
    return fig


def build_overlay_figure(source_path: Path | None, heatmap_path: Path | None) -> go.Figure:
    """Build image and heatmap overlay figure."""

    image = _load_image_array(source_path) if source_path is not None else None
    heatmap = _load_heatmap(heatmap_path) if heatmap_path is not None else None
    if image is None or heatmap is None:
        return _blank_figure("overlay")
    heatmap = _resize_nearest(heatmap, image.shape[0], image.shape[1])
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    fig.add_trace(go.Heatmap(z=heatmap, colorscale="Inferno", opacity=0.45, showscale=True))
    fig.update_layout(title="Image + heatmap overlay", margin={"l": 8, "r": 8, "t": 42, "b": 8})
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def _dropdown(dcc: Any, element_id: str, label: str) -> Any:
    return dcc.Dropdown(id=element_id, placeholder=label, clearable=False)


def _header_style() -> dict[str, str]:
    return {
        "padding": "24px 28px 16px",
        "backgroundColor": "white",
        "borderBottom": "1px solid #d8dee9",
    }


def _controls_style() -> dict[str, str]:
    return {
        "display": "grid",
        "gridTemplateColumns": "repeat(4, minmax(180px, 1fr)) 44px",
        "gap": "12px",
        "marginBottom": "16px",
        "alignItems": "end",
    }


def _help_button_style() -> dict[str, str]:
    return {
        "height": "38px",
        "border": "1px solid #2563eb",
        "borderRadius": "999px",
        "backgroundColor": "#2563eb",
        "color": "white",
        "fontWeight": "700",
        "fontSize": "18px",
        "cursor": "pointer",
    }


def _close_button_style() -> dict[str, str]:
    return {
        "border": "1px solid #c6ceda",
        "borderRadius": "6px",
        "backgroundColor": "white",
        "padding": "8px 12px",
        "cursor": "pointer",
    }


def _modal_overlay_style(hidden: bool) -> dict[str, Any]:
    return {
        "display": "none" if hidden else "flex",
        "position": "fixed",
        "inset": "0",
        "zIndex": 1000,
        "backgroundColor": "rgba(15, 23, 42, 0.45)",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "24px",
    }


def _modal_content_style() -> dict[str, str]:
    return {
        "backgroundColor": "white",
        "borderRadius": "8px",
        "border": "1px solid #d8dee9",
        "boxShadow": "0 24px 60px rgba(15, 23, 42, 0.25)",
        "width": "min(920px, 96vw)",
        "maxHeight": "86vh",
        "overflowY": "auto",
        "padding": "20px",
    }


def _modal_header_style() -> dict[str, str]:
    return {
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "space-between",
        "gap": "12px",
        "marginBottom": "14px",
    }


def _grid_style() -> dict[str, str]:
    return {
        "display": "grid",
        "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
        "gap": "16px",
        "marginBottom": "18px",
    }


def _preview_grid_style() -> dict[str, str]:
    return {
        "display": "grid",
        "gridTemplateColumns": "minmax(240px, 0.8fr) repeat(3, minmax(0, 1fr))",
        "gap": "16px",
        "marginTop": "14px",
    }


def _detail_style() -> dict[str, str]:
    return {
        "backgroundColor": "white",
        "border": "1px solid #d8dee9",
        "padding": "12px",
        "overflow": "auto",
        "minHeight": "280px",
        "whiteSpace": "pre-wrap",
    }


def metric_help_for(metric_name: str) -> dict[str, Any]:
    """Return help content for a metric name.

    Parameters
    ----------
    metric_name
        Metric instance name or class-like name.

    Returns
    -------
    dict[str, Any]
        Help content used by the Dash modal.
    """

    normalized = metric_name.lower()
    for key, value in _metric_help_catalog().items():
        if key.lower() in normalized:
            return value
    return _metric_help_catalog()["default"]


def _metric_help_component(html: Any, help_data: dict[str, Any]) -> Any:
    stats = help_data.get("stats", {})
    thresholds = help_data.get("thresholds", [])
    notes = help_data.get("notes", [])
    return html.Div(
        [
            html.H3(str(help_data["title"]), style={"marginTop": "0"}),
            html.P(str(help_data["purpose"])),
            html.H4("主なstat"),
            html.Ul(
                [
                    html.Li([html.Code(stat), ": ", description])
                    for stat, description in stats.items()
                ]
            ),
            html.H4("見方の目安"),
            html.Ul([html.Li(item) for item in thresholds]),
            html.H4("注意点"),
            html.Ul([html.Li(item) for item in notes]),
        ],
        style={"lineHeight": "1.6"},
    )


def _metric_help_catalog() -> dict[str, dict[str, Any]]:
    return {
        "tensor": {
            "title": "TensorDiffMetric",
            "purpose": "raw tensor同士の数値差分を確認するmetricです。",
            "stats": {
                "shape_match_rate": "shape一致率。1.0であることが前提です。",
                "cosine_similarity": "ベクトル方向の一致度。1.0に近いほど良いです。",
                "mean_abs_error": "全要素の平均絶対誤差。全体的なズレ量を見ます。",
                "max_abs_error": "最大絶対誤差。一部だけ大きくズレていないかを見ます。",
                "rmse": "二乗平均平方根誤差。大きなズレをやや強く反映します。",
                "allclose_rate": "atol/rtol条件で一致した要素の割合です。",
            },
            "thresholds": [
                "logits: cosine_similarity >= 0.9999 をまず目安にします。",
                "mean_abs_error / max_abs_error は出力スケールに依存します。",
                "FP16では完全一致ではなく、タスク出力が保たれているかを重視します。",
            ],
            "notes": [
                "raw metricは数値差分を見るもので、最終判断はpost metricを優先します。",
                "shapeが一致しない場合は変換やmappingの問題を先に確認してください。",
            ],
        },
        "feature": {
            "title": "FeatureMapDiffMetric",
            "purpose": "CNN backboneなどの中間feature mapの一致度を確認するmetricです。",
            "stats": {
                "layer_cosine_similarity": "layer全体のcosine類似度です。",
                "mean_channel_cosine_similarity": "channelごとのcosine類似度の平均です。",
                "min_channel_cosine_similarity": "もっとも悪いchannelのcosine類似度です。",
                "spatial_mean_abs_error": "spatial位置ごとの平均絶対誤差です。",
                "worst_sample_id": "そのlayerで最大誤差が出たsampleです。",
                "heatmap_path": "保存されたspatial error heatmapのpathです。",
            },
            "thresholds": [
                "layer_cosine_similarity >= 0.999 を初期目安にします。",
                "深いlayerやタスクに近いlayerほどpost metricと合わせて確認します。",
            ],
            "notes": [
                "min_channel_cosine_similarityが低い場合、特定channelだけ劣化している可能性があります。",
                "heatmap previewで局所的なズレを確認できます。",
            ],
        },
        "classification": {
            "title": "Classification Metrics",
            "purpose": "分類結果の一致度やlabelありaccuracyを確認するmetricです。",
            "stats": {
                "top1_match_rate": "referenceとtargetのtop-1予測一致率です。",
                "top5_match_rate": "top-5集合の一致率です。",
                "topk_ranking_match_rate": "ranking全体の一致率です。",
                "kl_divergence": "reference分布からtarget分布へのKL divergenceです。",
                "js_divergence": "対称的に見やすいJS divergenceです。",
                "top1_accuracy": "labelに対するtop-1 accuracyです。",
            },
            "thresholds": [
                "変換精度確認では top1_match_rate >= 0.99 を初期目安にします。",
                "JS divergenceは0に近いほど良いです。",
            ],
            "notes": [
                "reference自体が間違っていても一致は高くなるので、labelありaccuracyも別途見ると安心です。",
                "logits比較よりpost metricの一致を最終判断に寄せます。",
            ],
        },
        "detection": {
            "title": "Detection Metrics",
            "purpose": "検出結果のbox、class、score、簡易AP/mAPを確認するmetricです。",
            "stats": {
                "mean_iou": "対応付けられたboxの平均IoUです。",
                "match_rate": "reference boxに対してtargetが対応付いた割合です。",
                "class_match_rate": "対応box同士のclass一致率です。",
                "mean_confidence_abs_diff": "対応box同士のconfidence差分平均です。",
                "box_count_diff": "target box数 - reference box数です。",
                "map": "指定IoU閾値群に対する簡易mAPです。",
            },
            "thresholds": [
                "class_match_rate >= 0.99 を初期目安にします。",
                "mAP deltaは用途により 0.001〜0.005 程度から確認します。",
                "box_count_diffが大きい場合はNMSやscore threshold差を確認します。",
            ],
            "notes": [
                "DetectionConsistencyMetricはreference vs targetの一致性を見ます。",
                "DetectionAccuracyMetricはannotationsまたはreferenceをGT扱いしてAPを計算します。",
            ],
        },
        "default": {
            "title": "Metric Help",
            "purpose": (
                "選択中metricの説明が未登録です。raw metricは数値差分、"
                "post metricはタスク出力の一致を確認します。"
            ),
            "stats": {
                "value": "metricが出力した値です。意味はmetric実装またはREADMEを確認してください。",
            },
            "thresholds": [
                "raw metricは完全一致ではなく許容範囲を見ます。",
                "最終判断ではpost metricを優先します。",
            ],
            "notes": [
                "独自metricを追加した場合はhelp catalogへ説明を追加すると運用しやすくなります。",
            ],
        },
    }


def _options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def _select_value(current: str | None, values: list[str]) -> str:
    if current in values:
        return str(current)
    return values[0] if values else ""


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in result:
            result.append(text)
    return result


def _filter_report(rows: list[dict[str, Any]], report_value: str) -> list[dict[str, Any]]:
    if report_value == "all":
        return rows
    return [row for row in rows if row.get("report") == report_value]


def _row_in_report(row: dict[str, Any], report_value: str) -> bool:
    return report_value == "all" or row.get("report") == report_value


def _columns(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in keys:
                keys.append(key)
    return [{"name": key, "id": key} for key in keys]


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _row_stat_label(row: dict[str, Any]) -> str:
    return f"{row.get('output') or '_'} / {row.get('stat') or '_'}"


def _selected_row(
    selected_rows: list[int] | None,
    table_rows: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not table_rows:
        return None
    if selected_rows:
        index = selected_rows[0]
        if 0 <= index < len(table_rows):
            return table_rows[index]
    return table_rows[0]


def _first_row_for_preview(
    rows: list[dict[str, Any]],
    report_value: str,
    comparison_value: str,
    stage_value: str,
    metric_value: str,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            _row_in_report(row, report_value)
            and row.get("comparison") == comparison_value
            and row.get("stage") == stage_value
            and row.get("metric") == metric_value
        ):
            return row
    return None


def _heatmap_path_for_row(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> Path | None:
    if row.get("stat") == "heatmap_path":
        return _path_from_row(row, "value")
    for candidate in rows:
        if (
            candidate.get("sample_id") == row.get("sample_id")
            and candidate.get("comparison") == row.get("comparison")
            and candidate.get("metric") == row.get("metric")
            and candidate.get("stat") == "heatmap_path"
        ):
            return _path_from_row(candidate, "value")
    return None


def _path_from_row(row: dict[str, Any], key: str) -> Path | None:
    value = row.get(key)
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    report_dir = row.get("_report_dir")
    if report_dir:
        candidate = Path(str(report_dir)) / path
        if candidate.exists():
            return candidate
    return path


def _detail_text(row: dict[str, Any], heatmap_path: Path | None, source_path: Path | None) -> str:
    visible = {key: value for key, value in row.items() if not key.startswith("_")}
    visible["resolved_source_path"] = str(source_path) if source_path else ""
    visible["resolved_heatmap_path"] = str(heatmap_path) if heatmap_path else ""
    import json

    return json.dumps(visible, ensure_ascii=False, indent=2, default=str)


def _load_image_array(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except OSError:
        return None


def _load_heatmap(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        heatmap = np.load(path)
    except (OSError, ValueError):
        return None
    if heatmap.ndim > 2:
        heatmap = np.squeeze(heatmap)
    if heatmap.ndim != 2:
        return None
    return np.asarray(heatmap, dtype=np.float64)


def _resize_nearest(array: np.ndarray, height: int, width: int) -> np.ndarray:
    y_indices = np.linspace(0, array.shape[0] - 1, height).astype(int)
    x_indices = np.linspace(0, array.shape[1] - 1, width).astype(int)
    resized: np.ndarray = array[y_indices][:, x_indices]
    return resized


def _blank_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, annotations=[{"text": "No preview", "showarrow": False}])
    return fig
