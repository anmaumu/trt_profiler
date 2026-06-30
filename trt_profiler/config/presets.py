"""Preset resolution for concise configs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from trt_profiler.config.schema import (
    ComponentConfig,
    EvaluationConfig,
    InputConfig,
    MetricConfig,
    ReportConfig,
    VariantConfig,
)
from trt_profiler.core.types import ConfigDict

METRIC_PRESETS = {
    "tensor_diff": "trt_profiler.metrics.TensorDiffMetric",
    "feature_diff": "trt_profiler.metrics.FeatureMapDiffMetric",
    "classification_consistency": "trt_profiler.metrics.ClassificationConsistencyMetric",
    "classification_accuracy": "trt_profiler.metrics.ClassificationAccuracyMetric",
    "detection_consistency": "trt_profiler.metrics.DetectionConsistencyMetric",
    "detection_accuracy": "trt_profiler.metrics.DetectionAccuracyMetric",
}

POSTPROCESSOR_PRESETS = {
    "identity": "trt_profiler.postprocessors.IdentityPostprocessor",
    "softmax": "trt_profiler.postprocessors.SoftmaxPostprocessor",
}

REPORTER_PRESETS = {
    "json": ("trt_profiler.report.JsonReporter", "path", "report.json"),
    "dashboard": ("trt_profiler.report.PlotlyDashboardReporter", "path", "dashboard.html"),
    "html": ("trt_profiler.report.PlotlyDashboardReporter", "path", "dashboard.html"),
    "csv": ("trt_profiler.report.CsvReporter", "output_dir", "csv"),
}

VariantFactory = Callable[[VariantConfig, str, EvaluationConfig], ConfigDict]


def build_dataset(input_config: InputConfig) -> ConfigDict:
    """Build the full dataset config.

    Parameters
    ----------
    input_config
        Parsed input config.

    Returns
    -------
    ConfigDict
        Full dataset config consumed by the pipeline.
    """

    if input_config.kind in {"npz", "npz_folder"}:
        return {
            "type": "npz_folder",
            "config": {
                "path": input_config.path,
                "pattern": input_config.pattern,
                **input_config.options,
            },
        }
    if input_config.kind in {"image", "image_folder"}:
        return {
            "type": "image_folder",
            "config": {"path": input_config.path, **input_config.options},
        }
    if input_config.kind in {"video", "video_file", "video_folder"}:
        return {
            "type": input_config.kind,
            "config": {"path": input_config.path, **input_config.options},
        }
    return {
        "type": input_config.kind,
        "config": {"path": input_config.path, **input_config.options},
    }


def resolve_preprocess(config: ComponentConfig | None, input_config: InputConfig) -> ConfigDict:
    """Resolve the preprocessor config.

    Parameters
    ----------
    config
        Parsed preprocessor component.
    input_config
        Parsed input config used for defaults.

    Returns
    -------
    ConfigDict
        Full preprocessor config.
    """

    if config is not None and config.class_path is not None:
        return {"class": config.class_path, "config": config.config}

    preset = config.preset if config is not None else input_config.kind
    if preset is None:
        preset = input_config.kind
    preset = preset.lower()
    component_config = dict(config.config) if config is not None else {}
    if preset in {"npz", "npz_folder"}:
        return {
            "class": "trt_profiler.preprocessors.NpzPreprocessor",
            "config": {
                "inputs": [
                    {
                        "name": input_config.input_name,
                        "npz_key": input_config.npz_key,
                        "dtype": input_config.dtype,
                    }
                ]
            },
        }
    if preset in {"imagenet", "image", "image_folder", "video"}:
        component_config.setdefault("input_name", input_config.input_name)
        return {
            "class": "trt_profiler.preprocessors.ImageNetPreprocessor",
            "config": component_config,
        }
    raise ValueError(f"Unsupported preprocess preset: {preset!r}")


def resolve_postprocessor(component: ComponentConfig) -> ConfigDict:
    """Resolve a postprocessor config."""

    if component.class_path is not None:
        resolved = {"class": component.class_path, "config": component.config}
        if component.name is not None:
            resolved["name"] = component.name
        return resolved

    preset = component.preset or "identity"
    class_path = POSTPROCESSOR_PRESETS.get(preset)
    if class_path is None:
        raise ValueError(f"Unsupported postprocess preset: {preset!r}")
    return {"name": component.name or preset, "class": class_path, "config": component.config}


def resolve_metric(metric: MetricConfig) -> ConfigDict:
    """Resolve a metric config."""

    if metric.class_path is not None:
        return {"name": metric.name, "class": metric.class_path, "config": metric.config}

    preset = metric.preset or metric.name
    class_path = METRIC_PRESETS.get(preset)
    if class_path is None:
        raise ValueError(f"Unsupported metric preset: {preset!r}")
    return {"name": metric.name, "class": class_path, "config": metric.config}


def resolve_reporter(format_name: str, report: ReportConfig) -> ConfigDict:
    """Resolve one reporter format preset."""

    reporter = REPORTER_PRESETS.get(format_name)
    if reporter is None:
        raise ValueError(f"Unsupported report format: {format_name!r}")
    class_path, path_key, relative_path = reporter
    return {
        "class": class_path,
        "config": {path_key: str(Path(report.output_dir) / relative_path)},
    }


def resolve_variant(variant: VariantConfig, role: str, config: EvaluationConfig) -> ConfigDict:
    """Resolve a backend variant.

    Parameters
    ----------
    variant
        Parsed variant config.
    role
        Default role computed from list position.
    config
        Top-level evaluation config.

    Returns
    -------
    ConfigDict
        Full variant config consumed by the pipeline.
    """

    if variant.builder is not None and variant.runner is not None:
        return {
            "name": variant.name,
            "role": variant.role or role,
            **variant.extra,
            "builder": variant.builder,
            "runner": variant.runner,
        }

    preset = variant.preset or variant.name
    factory = _variant_presets().get(preset)
    if factory is None:
        raise ValueError(f"Unsupported variant preset: {preset!r}")
    return factory(variant, role, config)


def _variant_presets() -> dict[str, VariantFactory]:
    return {
        "ort_cpu": lambda variant, role, config: _ort_variant(
            variant, role, ["CPUExecutionProvider"]
        ),
        "ort_cuda": lambda variant, role, config: _ort_variant(
            variant, role, ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ),
        "ort_trt": lambda variant, role, config: _ort_variant(
            variant,
            role,
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        "openvino_cpu": _openvino_cpu_variant,
        "trt_fp32": lambda variant, role, config: _trt_variant(variant, role, "fp32", config),
        "trt_fp16": lambda variant, role, config: _trt_variant(variant, role, "fp16", config),
    }


def _ort_variant(variant: VariantConfig, role: str, providers: list[str]) -> ConfigDict:
    return {
        "name": variant.name,
        "backend": "onnxruntime",
        "role": variant.role or role,
        "builder": {"class": "trt_profiler.builders.OnnxPassthroughBuilder", "config": {}},
        "runner": {
            "class": "trt_profiler.runners.OnnxRuntimeRunner",
            "config": {"providers": providers},
        },
    }


def _openvino_cpu_variant(
    variant: VariantConfig, role: str, config: EvaluationConfig
) -> ConfigDict:
    return {
        "name": variant.name,
        "backend": "openvino",
        "role": variant.role or role,
        "builder": {"class": "trt_profiler.builders.OpenVinoBuilder", "config": {}},
        "runner": {"class": "trt_profiler.runners.OpenVinoRunner", "config": {"device": "CPU"}},
    }


def _trt_variant(
    variant: VariantConfig, role: str, precision: str, config: EvaluationConfig
) -> ConfigDict:
    trt_config = dict(config.tensor_rt)
    artifacts_dir = str(trt_config.pop("artifacts_dir", "artifacts"))
    engine_path = str(
        trt_config.pop("engine_path", Path(artifacts_dir) / f"{config.model.name}_{precision}.plan")
    )
    return {
        "name": variant.name,
        "backend": "tensorrt",
        "role": variant.role or role,
        "precision": precision,
        "builder": {
            "class": "trt_profiler.builders.TensorRTBuilder",
            "config": {
                "build": True,
                "builder_backend": "trtexec",
                "engine_path": engine_path,
                "precision": precision,
                **trt_config,
            },
        },
        "runner": {"class": "trt_profiler.runners.TensorRTRunner", "config": {}},
    }
