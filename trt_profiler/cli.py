from __future__ import annotations

import argparse

from trt_profiler.config.loader import load_config
from trt_profiler.core.pipeline import EvaluationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="trt-profiler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run an accuracy evaluation.")
    eval_parser.add_argument("-c", "--config", required=True, help="Path to YAML config.")

    args = parser.parse_args()
    if args.command == "eval":
        config = load_config(args.config)
        EvaluationPipeline(config).run()
