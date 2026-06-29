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
    comparison_group = eval_parser.add_mutually_exclusive_group()
    comparison_group.add_argument(
        "--all-combinations",
        action="store_true",
        help="Evaluate all pairwise variant combinations defined in the config.",
    )
    comparison_group.add_argument(
        "--reference-to-all",
        action="store_true",
        help="Evaluate each reference variant against all other variants.",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Generate a Plotly HTML dashboard from a report JSON.",
    )
    dashboard_parser.add_argument("report_json", help="Path to report.json.")
    dashboard_parser.add_argument(
        "-o", "--output", required=True, help="Output dashboard HTML path."
    )

    dash_parser = subparsers.add_parser(
        "dash",
        help="Run a Dash server for one or more report JSON files.",
    )
    dash_parser.add_argument("report_json", nargs="+", help="Path(s) to report.json.")
    dash_parser.add_argument("--host", default="127.0.0.1", help="Dash server host.")
    dash_parser.add_argument("--port", default=8050, type=int, help="Dash server port.")

    args = parser.parse_args()
    if args.command == "eval":
        config = load_config(args.config)
        comparison_mode = _comparison_mode_from_args(args)
        EvaluationPipeline(config, comparison_mode=comparison_mode).run()
    elif args.command == "dashboard":
        report_data = load_report_data(args.report_json)
        PlotlyDashboardReporter(config={"path": args.output}).write(report_data)
    elif args.command == "dash":
        from trt_profiler.report.dash_app import run_dash_server

        run_dash_server(args.report_json, host=args.host, port=args.port)


def _comparison_mode_from_args(args: argparse.Namespace) -> str | None:
    """Resolve comparison mode from CLI arguments.

    Parameters
    ----------
    args
        Parsed CLI arguments.

    Returns
    -------
    str | None
        Pipeline comparison mode override, or ``None`` to use the config.
    """

    if getattr(args, "all_combinations", False):
        return "all-pairs"
    if getattr(args, "reference_to_all", False):
        return "reference-to-all"
    return None
