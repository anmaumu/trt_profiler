"""Command line interface for trt-profiler."""

from __future__ import annotations

import argparse

from trt_profiler.config.loader import load_config
from trt_profiler.core.pipeline import EvaluationPipeline
from trt_profiler.report.loader import load_report_data
from trt_profiler.report.plotly_dashboard import PlotlyDashboardReporter


def main() -> None:
    """Run the command line interface.

    Notes
    -----
    The CLI currently provides ``eval`` for running an evaluation and
    ``dashboard`` for generating a Plotly HTML dashboard from an existing JSON
    report.
    """

    parser = argparse.ArgumentParser(prog="trt-profiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run an accuracy evaluation.")
    eval_parser.add_argument("-c", "--config", required=True, help="Path to YAML config.")

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Generate a Plotly HTML dashboard from a report JSON.",
    )
    dashboard_parser.add_argument("report_json", help="Path to report.json.")
    dashboard_parser.add_argument(
        "-o", "--output", required=True, help="Output dashboard HTML path."
    )

    args = parser.parse_args()
    if args.command == "eval":
        config = load_config(args.config)
        EvaluationPipeline(config).run()
    elif args.command == "dashboard":
        report_data = load_report_data(args.report_json)
        PlotlyDashboardReporter(config={"path": args.output}).write(report_data)
