"""Build pipeline config dictionaries from readable config dataclasses."""

from __future__ import annotations

from trt_profiler.config.presets import (
    build_dataset,
    resolve_metric,
    resolve_postprocessor,
    resolve_preprocess,
    resolve_reporter,
    resolve_variant,
)
from trt_profiler.config.schema import EvaluationConfig
from trt_profiler.core.types import ConfigDict


def build_pipeline_config(config: EvaluationConfig) -> ConfigDict:
    """Build the full pipeline configuration.

    Parameters
    ----------
    config
        Parsed readable configuration.

    Returns
    -------
    ConfigDict
        Full dict consumed by ``EvaluationPipeline``.
    """

    variants = [
        resolve_variant(variant, "reference" if index == 0 else "target", config)
        for index, variant in enumerate(config.variants)
    ]
    variant_names = [str(variant["name"]) for variant in variants]

    return {
        "version": 1,
        "common": {
            "model": {
                "name": config.model.name,
                "source_path": config.model.source_path,
                "format": config.model.format,
                "tasks": config.model.tasks,
            },
            "variants": variants,
            "dataset": build_dataset(config.input),
            "comparisons": config.comparisons
            or _explicit_comparisons(config.compare, variant_names),
            "comparison_mode": _comparison_mode(config.compare, variant_names),
            "input_mapping": {
                name: {config.input.backend_name: config.input.input_name} for name in variant_names
            },
            "output_mapping": {name: dict(config.outputs) for name in variant_names},
            "metrics": {
                "raw": [resolve_metric(metric) for metric in config.metrics.raw],
                "post": [resolve_metric(metric) for metric in config.metrics.post],
            },
            "report": {
                "output_dir": config.report.output_dir,
                "reporters": [
                    resolve_reporter(format_name, config.report)
                    for format_name in config.report.formats
                ],
            },
        },
        "preprocess": resolve_preprocess(config.preprocess, config.input),
        "postprocessors": [
            resolve_postprocessor(component) for component in config.postprocessors
        ],
    }


def _explicit_comparisons(compare: str, variant_names: list[str]) -> list[ConfigDict]:
    if compare != "configured" or len(variant_names) < 2:
        return []
    return [
        {
            "name": f"{variant_names[0]}_vs_{variant_names[1]}",
            "reference": variant_names[0],
            "target": variant_names[1],
        }
    ]


def _comparison_mode(compare: str, variant_names: list[str]) -> str:
    aliases = {"all-combinations": "all-pairs", "first-to-all": "reference-to-all"}
    mode = aliases.get(compare, compare)
    if mode in {"all-pairs", "reference-to-all", "configured"}:
        return mode
    if len(variant_names) > 2:
        return "reference-to-all"
    return "configured"
