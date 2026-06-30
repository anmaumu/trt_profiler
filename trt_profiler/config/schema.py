"""Readable configuration schema used by the loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trt_profiler.core.types import ConfigDict


@dataclass(frozen=True)
class ModelConfig:
    """Source model configuration.

    Parameters
    ----------
    name
        Human-readable model name.
    source_path
        Source model path.
    format
        Source model format.
    tasks
        Optional task labels such as ``classification`` or ``detection``.
    """

    name: str
    source_path: str
    format: str = "onnx"
    tasks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InputConfig:
    """Input dataset and model input mapping configuration.

    Parameters
    ----------
    kind
        Dataset shorthand such as ``npz``, ``image_folder``, or ``video``.
    path
        Dataset path.
    input_name
        Common input name produced by preprocessors.
    backend_name
        Backend-visible model input name.
    pattern
        File glob used by folder datasets.
    npz_key
        NPZ array key used by the default NPZ preprocessor.
    dtype
        Input dtype used by the default NPZ preprocessor.
    options
        Dataset loader options not interpreted by the config parser.
    """

    kind: str
    path: str
    input_name: str = "input_tensor"
    backend_name: str = "data"
    pattern: str = "*.npz"
    npz_key: str = "data"
    dtype: str = "float32"
    options: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class ComponentConfig:
    """Configurable class or preset component.

    Parameters
    ----------
    name
        Optional component name.
    preset
        Preset name such as ``softmax`` or ``npz``.
    class_path
        Fully qualified custom class path.
    config
        Component-specific configuration.
    """

    name: str | None = None
    preset: str | None = None
    class_path: str | None = None
    config: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class MetricConfig:
    """Metric preset or custom metric configuration.

    Parameters
    ----------
    name
        Metric name used in reports.
    preset
        Metric preset name.
    class_path
        Fully qualified custom metric class path.
    config
        Metric-specific configuration.
    """

    name: str
    preset: str | None = None
    class_path: str | None = None
    config: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class VariantConfig:
    """Backend variant configuration.

    Parameters
    ----------
    name
        Variant name.
    preset
        Variant preset such as ``ort_cpu`` or ``trt_fp16``.
    role
        Optional comparison role.
    builder
        Full builder config when not using a preset.
    runner
        Full runner config when not using a preset.
    extra
        Additional fields preserved for full custom variants.
    """

    name: str
    preset: str | None = None
    role: str | None = None
    builder: ConfigDict | None = None
    runner: ConfigDict | None = None
    extra: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class MetricsConfig:
    """Metric groups.

    Parameters
    ----------
    raw
        Metrics evaluated on raw runner outputs.
    post
        Metrics evaluated on postprocessed outputs.
    """

    raw: list[MetricConfig] = field(default_factory=list)
    post: list[MetricConfig] = field(default_factory=list)


@dataclass(frozen=True)
class ReportConfig:
    """Report output configuration.

    Parameters
    ----------
    output_dir
        Directory where default reporters write files.
    formats
        Reporter format presets.
    """

    output_dir: str = "reports"
    formats: list[str] = field(default_factory=lambda: ["json", "dashboard"])


@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level readable evaluation configuration.

    Parameters
    ----------
    model
        Source model configuration.
    input
        Dataset and input mapping configuration.
    outputs
        Logical output name to backend output name mapping.
    variants
        Backend variants to evaluate.
    comparisons
        Optional explicit reference-target comparison pairs.
    metrics
        Raw and postprocess metric groups.
    preprocess
        Preprocessor preset or custom class.
    postprocessors
        Postprocessor presets or custom classes.
    report
        Report output settings.
    compare
        Comparison mode shorthand.
    tensor_rt
        TensorRT build defaults shared by TensorRT presets.
    """

    model: ModelConfig
    input: InputConfig
    outputs: dict[str, str]
    variants: list[VariantConfig]
    comparisons: list[ConfigDict] = field(default_factory=list)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    preprocess: ComponentConfig | None = None
    postprocessors: list[ComponentConfig] = field(default_factory=list)
    report: ReportConfig = field(default_factory=ReportConfig)
    compare: str = "configured"
    tensor_rt: ConfigDict = field(default_factory=dict)


RawConfig = dict[str, Any]


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    """Require keys in a configuration mapping.

    Parameters
    ----------
    config
        Configuration mapping to validate.
    keys
        Required key names.
    context
        Human-readable context used in error messages.

    Raises
    ------
    ValueError
        If one or more keys are missing.
    """

    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {missing}")
