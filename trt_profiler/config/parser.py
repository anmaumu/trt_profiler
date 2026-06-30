"""Parse YAML mappings into readable config dataclasses."""

from __future__ import annotations

from typing import Any

from trt_profiler.config.schema import (
    ComponentConfig,
    EvaluationConfig,
    InputConfig,
    MetricConfig,
    MetricsConfig,
    ModelConfig,
    ReportConfig,
    VariantConfig,
)
from trt_profiler.core.types import ConfigDict


def parse_evaluation_config(raw: ConfigDict) -> EvaluationConfig:
    """Parse a concise YAML mapping.

    Parameters
    ----------
    raw
        Concise configuration mapping.

    Returns
    -------
    EvaluationConfig
        Parsed readable configuration.
    """

    return EvaluationConfig(
        model=_parse_model(_dict(raw.get("model"))),
        input=_parse_input(_dict(raw.get("input", raw.get("dataset")))),
        outputs=_str_dict(_dict(raw.get("outputs"))),
        variants=_parse_variants(raw.get("variants", [])),
        comparisons=[dict(item) for item in raw.get("comparisons", []) if isinstance(item, dict)],
        metrics=_parse_metrics(_dict(raw.get("metrics"))),
        preprocess=_parse_component(raw.get("preprocess"), default_preset=None),
        postprocessors=_parse_postprocessors(raw.get("postprocess", raw.get("postprocessors"))),
        report=_parse_report(_dict(raw.get("report"))),
        compare=str(raw.get("compare", raw.get("comparison_mode", "configured"))),
        tensor_rt=_dict(raw.get("tensor_rt", raw.get("tensorrt"))),
    )


def _parse_model(raw: ConfigDict) -> ModelConfig:
    source_path = raw.get("path", raw.get("source_path"))
    return ModelConfig(
        name=str(raw["name"]),
        source_path=str(source_path),
        format=str(raw.get("format", "onnx")),
        tasks=[str(item) for item in raw.get("tasks", [])],
    )


def _parse_input(raw: ConfigDict) -> InputConfig:
    parser_keys = {
        "type",
        "path",
        "input_name",
        "backend_name",
        "model_input",
        "pattern",
        "npz_key",
        "dtype",
    }
    return InputConfig(
        kind=str(raw.get("type", "npz_folder")),
        path=str(raw["path"]),
        input_name=str(raw.get("input_name", "input_tensor")),
        backend_name=str(raw.get("backend_name", raw.get("model_input", "data"))),
        pattern=str(raw.get("pattern", "*.npz")),
        npz_key=str(raw.get("npz_key", "data")),
        dtype=str(raw.get("dtype", "float32")),
        options={str(key): value for key, value in raw.items() if key not in parser_keys},
    )


def _parse_variants(raw: Any) -> list[VariantConfig]:
    if not isinstance(raw, list):
        raise ValueError("variants must be a list.")
    return [_parse_variant(item) for item in raw]


def _parse_variant(item: Any) -> VariantConfig:
    if isinstance(item, str):
        return VariantConfig(name=item, preset=item)
    if not isinstance(item, dict):
        raise ValueError(f"Unsupported variant item: {item!r}")

    extra = {
        str(key): value
        for key, value in item.items()
        if key not in {"name", "preset", "role", "builder", "runner"}
    }
    name = str(item["name"])
    return VariantConfig(
        name=name,
        preset=str(item.get("preset", name)),
        role=str(item["role"]) if "role" in item else None,
        builder=_dict_or_none(item.get("builder")),
        runner=_dict_or_none(item.get("runner")),
        extra=extra,
    )


def _parse_metrics(raw: ConfigDict) -> MetricsConfig:
    return MetricsConfig(
        raw=[_parse_metric(item) for item in raw.get("raw", [])],
        post=[_parse_metric(item) for item in raw.get("post", [])],
    )


def _parse_metric(item: Any) -> MetricConfig:
    if isinstance(item, str):
        return MetricConfig(name=item, preset=item)
    if not isinstance(item, dict):
        raise ValueError(f"Unsupported metric item: {item!r}")
    if "class" in item:
        class_path = str(item["class"])
        return MetricConfig(
            name=str(item.get("name", class_path.rsplit(".", maxsplit=1)[-1])),
            class_path=class_path,
            config=_dict(item.get("config")),
        )
    if len(item) == 1:
        name, config = next(iter(item.items()))
        return MetricConfig(name=str(name), preset=str(name), config=_dict(config))
    raise ValueError(f"Unsupported metric item: {item!r}")


def _parse_postprocessors(raw: Any) -> list[ComponentConfig]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    return [_parse_required_component(item, default_preset="identity") for item in items]


def _parse_required_component(raw: Any, default_preset: str) -> ComponentConfig:
    component = _parse_component(raw, default_preset=default_preset)
    if component is None:
        raise ValueError("component config is required.")
    return component


def _parse_component(raw: Any, default_preset: str | None) -> ComponentConfig | None:
    if raw is None:
        if default_preset is None:
            return None
        return ComponentConfig(preset=default_preset)
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported component config: {raw!r}")
    if "class" in raw:
        return ComponentConfig(
            name=str(raw["name"]) if "name" in raw else None,
            preset=str(raw["type"]) if "type" in raw else None,
            class_path=str(raw["class"]),
            config=_dict(raw.get("config")),
        )

    preset = str(raw.get("type", default_preset)) if raw.get("type", default_preset) else None
    config = {str(key): value for key, value in raw.items() if key not in {"name", "type"}}
    return ComponentConfig(
        name=str(raw["name"]) if "name" in raw else None,
        preset=preset,
        config=config,
    )


def _parse_report(raw: ConfigDict) -> ReportConfig:
    return ReportConfig(
        output_dir=str(raw.get("output_dir", "reports")),
        formats=[str(item) for item in raw.get("formats", ["json", "dashboard"])],
    )


def _dict(value: Any) -> ConfigDict:
    return value if isinstance(value, dict) else {}


def _dict_or_none(value: Any) -> ConfigDict | None:
    return value if isinstance(value, dict) else None


def _str_dict(value: ConfigDict) -> dict[str, str]:
    return {str(key): str(item) for key, item in value.items()}
