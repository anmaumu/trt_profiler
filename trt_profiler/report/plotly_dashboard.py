"""Plotly HTML dashboard reporter."""

from __future__ import annotations

import json
from dataclasses import asdict
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

        Notes
        -----
        The generated dashboard is a static HTML file. It uses Plotly.js in the
        browser, so users can select comparison pairs, stages, and metrics
        without running a Dash server.
        """

        output_path = Path(str(self.config.get("path", "dashboard.html")))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(report_data), encoding="utf-8")


def _build_html(report_data: ReportData) -> str:
    data_json = _json_for_script(asdict(report_data))
    title = _dashboard_title(report_data.metadata)
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f8fb;
      color: #172033;
    }}
    body {{
      margin: 0;
    }}
    header {{
      padding: 24px 28px 16px;
      background: #ffffff;
      border-bottom: 1px solid #d8dee9;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .subtitle {{
      color: #5f6b7a;
      font-size: 13px;
    }}
    main {{
      padding: 20px 28px 28px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: #445064;
      font-weight: 600;
    }}
    select {{
      min-height: 36px;
      border: 1px solid #c6ceda;
      border-radius: 6px;
      background: #ffffff;
      color: #172033;
      padding: 6px 10px;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d8dee9;
      border-radius: 8px;
      min-width: 0;
      padding: 10px;
    }}
    .panel.wide {{
      grid-column: 1 / -1;
    }}
    .plot {{
      width: 100%;
      height: 420px;
    }}
    .plot.short {{
      height: 320px;
    }}
    .section-title {{
      font-size: 15px;
      font-weight: 700;
      color: #172033;
      margin: 18px 0 10px;
    }}
    @media (max-width: 920px) {{
      .controls, .grid {{
        grid-template-columns: 1fr;
      }}
      .panel.wide {{
        grid-column: auto;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="subtitle" id="variantText"></div>
  </header>
  <main>
    <div class="section-title">全比較ペアの横断ビュー</div>
    <section class="grid">
      <div class="panel">
        <div id="comparisonMatrix" class="plot"></div>
      </div>
      <div class="panel">
        <div id="metricRanking" class="plot"></div>
      </div>
    </section>
    <div class="section-title">比較ペア詳細</div>
    <section class="controls">
      <label>比較ペア
        <select id="comparisonSelect"></select>
      </label>
      <label>ステージ
        <select id="stageSelect"></select>
      </label>
      <label>Metric
        <select id="metricSelect"></select>
      </label>
    </section>
    <section class="grid">
      <div class="panel">
        <div id="summaryBar" class="plot"></div>
      </div>
      <div class="panel">
        <div id="samplePlot" class="plot"></div>
      </div>
      <div class="panel wide">
        <div id="summaryTable" class="plot short"></div>
      </div>
      <div class="panel wide">
        <div id="failedTable" class="plot short"></div>
      </div>
    </section>
  </main>
  <script>
    const reportData = {data_json};
    const summaryRows = reportData.tables.metric_summary || [];
    const sampleRows = reportData.tables.per_sample || [];
    const failedRows = reportData.tables.failed_cases || [];

    const comparisonSelect = document.getElementById("comparisonSelect");
    const stageSelect = document.getElementById("stageSelect");
    const metricSelect = document.getElementById("metricSelect");
    const variantText = document.getElementById("variantText");

    const model = reportData.metadata.model || {{}};
    const variants = reportData.metadata.variants || [];
    variantText.textContent =
      `model: ${{model.name || "model"}} / variants: ${{variants.join(", ")}}`;

    function unique(values) {{
      return [...new Set(
        values.filter((value) => value !== undefined && value !== null && value !== "")
      )];
    }}

    function numeric(value) {{
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }}

    function fillSelect(select, values, fallback) {{
      const current = select.value;
      select.innerHTML = "";
      const items = values.length ? values : [fallback];
      for (const value of items) {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }}
      if (items.includes(current)) {{
        select.value = current;
      }}
    }}

    function allRows() {{
      return [...summaryRows, ...sampleRows];
    }}

    function refreshSelectors(changed) {{
      const comparisons = unique(allRows().map((row) => row.comparison));
      fillSelect(comparisonSelect, comparisons, "all");

      const comparison = comparisonSelect.value;
      const rowsForComparison = allRows().filter((row) => row.comparison === comparison);
      fillSelect(stageSelect, unique(rowsForComparison.map((row) => row.stage)), "raw");

      const stage = stageSelect.value;
      const rowsForStage = rowsForComparison.filter((row) => row.stage === stage);
      fillSelect(metricSelect, unique(rowsForStage.map((row) => row.metric)), "metric");

      if (changed === "comparison") {{
        stageSelect.selectedIndex = 0;
        const stageRows = rowsForComparison.filter((row) => row.stage === stageSelect.value);
        fillSelect(metricSelect, unique(stageRows.map((row) => row.metric)), "metric");
      }}
      if (changed === "stage") {{
        metricSelect.selectedIndex = 0;
      }}
    }}

    function filteredRows(rows) {{
      const comparison = comparisonSelect.value;
      const stage = stageSelect.value;
      const metric = metricSelect.value;
      return rows.filter((row) =>
        row.comparison === comparison && row.stage === stage && row.metric === metric
      );
    }}

    function matrixRows() {{
      const stage = stageSelect.value;
      const metric = metricSelect.value;
      return summaryRows.filter((row) => row.stage === stage && row.metric === metric);
    }}

    function renderComparisonMatrix(rows) {{
      const numericRows = rows.filter((row) => numeric(row.value) !== null);
      const comparisons = unique(numericRows.map((row) => row.comparison));
      const stats = unique(numericRows.map((row) => `${{row.output || "_"}} / ${{row.stat}}`));
      const z = stats.map((stat) =>
        comparisons.map((comparison) => {{
          const row = numericRows.find((item) =>
            item.comparison === comparison && `${{item.output || "_"}} / ${{item.stat}}` === stat
          );
          return row ? numeric(row.value) : null;
        }})
      );
      const trace = {{
        type: "heatmap",
        x: comparisons,
        y: stats,
        z,
        colorscale: "Blues",
        hovertemplate: "%{{x}}<br>%{{y}}<br>%{{z:.6g}}<extra></extra>",
        colorbar: {{ title: "value" }},
      }};
      Plotly.react("comparisonMatrix", [trace], {{
        title: "Comparison matrix",
        margin: {{ l: 150, r: 20, t: 48, b: 110 }},
        xaxis: {{ tickangle: -35, automargin: true }},
        yaxis: {{ automargin: true }},
      }}, {{ responsive: true }});
    }}

    function renderMetricRanking(rows) {{
      const numericRows = rows
        .filter((row) => numeric(row.value) !== null)
        .map((row) => ({{
          label: `${{row.comparison}} / ${{row.output || "_"}} / ${{row.stat}}`,
          value: numeric(row.value),
        }}))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 25)
        .reverse();
      const trace = {{
        type: "bar",
        orientation: "h",
        x: numericRows.map((row) => row.value),
        y: numericRows.map((row) => row.label),
        marker: {{ color: "#0f766e" }},
        hovertemplate: "%{{y}}<br>%{{x:.6g}}<extra></extra>",
      }};
      Plotly.react("metricRanking", [trace], {{
        title: "Metric ranking",
        margin: {{ l: 190, r: 18, t: 48, b: 40 }},
        xaxis: {{ zeroline: true }},
        yaxis: {{ automargin: true }},
      }}, {{ responsive: true }});
    }}

    function renderSummaryBar(rows) {{
      const numericRows = rows.filter((row) => numeric(row.value) !== null);
      const x = numericRows.map((row) => `${{row.output || "_"}} / ${{row.stat}}`);
      const y = numericRows.map((row) => numeric(row.value));
      const trace = {{
        type: "bar",
        x,
        y,
        marker: {{ color: "#2563eb" }},
        hovertemplate: "%{{x}}<br>%{{y:.6g}}<extra></extra>",
      }};
      Plotly.react("summaryBar", [trace], {{
        title: "Summary metrics",
        margin: {{ l: 56, r: 18, t: 48, b: 120 }},
        xaxis: {{ tickangle: -35, automargin: true }},
        yaxis: {{ zeroline: true }},
      }}, {{ responsive: true }});
    }}

    function renderSamplePlot(rows) {{
      const numericRows = rows.filter((row) => numeric(row.value) !== null);
      const grouped = {{}};
      for (const row of numericRows) {{
        const key = `${{row.output || "_"}} / ${{row.stat}}`;
        if (!grouped[key]) grouped[key] = {{ x: [], y: [], text: [] }};
        grouped[key].x.push(row.sample_id || "");
        grouped[key].y.push(numeric(row.value));
        grouped[key].text.push(row.sample_id || "");
      }}
      const traces = Object.entries(grouped).map(([name, data]) => ({{
        type: "box",
        name,
        y: data.y,
        boxpoints: "all",
        jitter: 0.35,
        pointpos: 0,
        text: data.text,
        hovertemplate: "%{{text}}<br>%{{y:.6g}}<extra>" + name + "</extra>",
      }}));
      Plotly.react("samplePlot", traces, {{
        title: "Per-sample distribution",
        margin: {{ l: 56, r: 18, t: 48, b: 110 }},
        xaxis: {{ tickangle: -35, automargin: true }},
      }}, {{ responsive: true }});
    }}

    function renderTable(divId, rows, columns, title) {{
      const displayRows = rows.slice(0, 300);
      const values = columns.map((column) => displayRows.map((row) => String(row[column] ?? "")));
      const trace = {{
        type: "table",
        header: {{
          values: columns,
          fill: {{ color: "#1f2937" }},
          font: {{ color: "white", size: 12 }},
          align: "left",
        }},
        cells: {{
          values,
          fill: {{ color: "#f8fafc" }},
          font: {{ color: "#111827", size: 11 }},
          align: "left",
          height: 24,
        }},
      }};
      Plotly.react(divId, [trace], {{
        title,
        margin: {{ l: 8, r: 8, t: 42, b: 8 }},
      }}, {{ responsive: true }});
    }}

    function render() {{
      const summary = filteredRows(summaryRows);
      const samples = filteredRows(sampleRows);
      const failed = filteredRows(failedRows);
      const crossComparison = matrixRows();
      renderComparisonMatrix(crossComparison);
      renderMetricRanking(crossComparison);
      renderSummaryBar(summary);
      renderSamplePlot(samples);
      renderTable(
        "summaryTable",
        summary,
        ["comparison", "stage", "metric", "output", "stat", "value"],
        "Summary table"
      );
      renderTable(
        "failedTable",
        failed,
        ["sample_id", "comparison", "stage", "metric", "output", "stat", "value", "status"],
        "Failed cases"
      );
    }}

    comparisonSelect.addEventListener("change", () => {{
      refreshSelectors("comparison");
      render();
    }});
    stageSelect.addEventListener("change", () => {{
      refreshSelectors("stage");
      render();
    }});
    metricSelect.addEventListener("change", render);

    refreshSelectors();
    render();
  </script>
</body>
</html>
"""


def _dashboard_title(metadata: dict[str, Any]) -> str:
    model = metadata.get("model", {})
    model_name = model.get("name", "model") if isinstance(model, dict) else "model"
    return f"TRT Profiler Dashboard - {model_name}"


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).replace("</", "<\\/")
