"""Shared data types and extension contracts.

The abstract base classes in this module define the contracts used by dataset
loaders, preprocessors, postprocessors, builders, runners, metrics, and
reporters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import numpy.typing as npt

NDArray: TypeAlias = npt.NDArray[Any]
TensorDict: TypeAlias = dict[str, NDArray]
ComparableDict: TypeAlias = dict[str, Any]
ConfigDict: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class TensorSpec:
    """Tensor metadata exposed by a runner.

    Parameters
    ----------
    name
        Backend-visible tensor name.
    shape
        Tensor shape. Dynamic dimensions may be represented by strings or
        ``None`` depending on the backend.
    dtype
        Backend dtype string.
    """

    name: str
    shape: tuple[int | str | None, ...]
    dtype: str


@dataclass(frozen=True)
class SourceModel:
    """Source model definition.

    Parameters
    ----------
    path
        Path to the source model file.
    format
        Source model format. The current main input format is ``"onnx"``.
    """

    path: Path
    format: str = "onnx"


@dataclass(frozen=True)
class ModelArtifact:
    """Runtime artifact produced for one backend variant.

    Parameters
    ----------
    variant_name
        Name of the backend variant that owns the artifact.
    backend
        Backend identifier such as ``"onnxruntime"``, ``"openvino"``, or
        ``"tensorrt"``.
    precision
        Optional precision label such as ``"fp32"`` or ``"fp16"``.
    path
        Path to the artifact consumed by a runner. Some builders may keep this
        value as ``None`` until a materialized artifact exists.
    config
        Builder-specific artifact metadata.
    """

    variant_name: str
    backend: str
    precision: str | None
    path: Path | None
    config: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class BackendVariant:
    """Configuration for one backend variant.

    Parameters
    ----------
    name
        Unique variant name used in comparisons and mappings.
    backend
        Backend identifier.
    role
        Logical role such as ``"reference"`` or ``"target"``.
    builder_class
        Dotted class path for the artifact builder.
    runner_class
        Dotted class path for the inference runner.
    builder_config
        Builder-specific configuration.
    runner_config
        Runner-specific configuration.
    precision
        Optional precision label.
    """

    name: str
    backend: str
    role: str
    builder_class: str
    runner_class: str
    builder_config: ConfigDict = field(default_factory=dict)
    runner_config: ConfigDict = field(default_factory=dict)
    precision: str | None = None


@dataclass(frozen=True)
class Comparison:
    """Reference-target comparison pair.

    Parameters
    ----------
    name
        Comparison name used in report rows.
    reference
        Variant name used as the baseline.
    target
        Variant name compared against the baseline.
    """

    name: str
    reference: str
    target: str


@dataclass(frozen=True)
class Sample:
    """Single dataset sample.

    Parameters
    ----------
    id
        Stable sample identifier used in per-sample report rows.
    data
        Loader-specific sample payload.
    source_path
        Optional original file path.
    label
        Optional classification or task label.
    annotations
        Optional structured annotations such as detection boxes.
    metadata
        Additional sample metadata.
    """

    id: str
    data: Any
    source_path: Path | None = None
    label: Any | None = None
    annotations: list[dict[str, Any]] = field(default_factory=list)
    metadata: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class SampleMetricRecord:
    """Per-sample metric record.

    Parameters
    ----------
    sample_id
        Evaluated sample identifier.
    comparison
        Comparison name.
    stage
        Evaluation stage, typically ``"raw"`` or ``"post"``.
    metric
        Metric name.
    stat
        Statistic name emitted by the metric.
    value
        Statistic value.
    output
        Optional output or layer name.
    threshold
        Optional pass/fail threshold.
    status
        Optional status such as ``"pass"`` or ``"fail"``.
    """

    sample_id: str
    comparison: str
    stage: str
    metric: str
    stat: str
    value: Any
    output: str | None = None
    threshold: float | None = None
    status: str | None = None


@dataclass(frozen=True)
class MetricSummaryRecord:
    """Aggregated metric record.

    Parameters
    ----------
    comparison
        Comparison name.
    stage
        Evaluation stage.
    metric
        Metric name.
    stat
        Aggregated statistic name.
    value
        Aggregated value.
    output
        Optional output or layer name.
    threshold
        Optional threshold associated with the statistic.
    status
        Optional aggregate status.
    """

    comparison: str
    stage: str
    metric: str
    stat: str
    value: Any
    output: str | None = None
    threshold: float | None = None
    status: str | None = None


@dataclass
class EvaluationResult:
    """Evaluation result before report formatting.

    Parameters
    ----------
    metadata
        Model, variant, and comparison metadata.
    summary
        Nested summary dictionary built from metric summary records.
    per_sample
        Flat list of per-sample metric records.
    """

    metadata: ConfigDict
    summary: ConfigDict
    per_sample: list[SampleMetricRecord] = field(default_factory=list)


@dataclass
class ReportData:
    """Report-ready data.

    Parameters
    ----------
    metadata
        Report metadata.
    summary
        Nested metric summary.
    tables
        Flat tables used by JSON and dashboard reporters.
    artifacts
        Extra report artifacts such as plot metadata or failed-case files.
    """

    metadata: ConfigDict
    summary: ConfigDict
    tables: dict[str, list[dict[str, Any]]]
    artifacts: ConfigDict = field(default_factory=dict)


class DatasetLoader(ABC):
    """Base class for iterable dataset loaders.

    Parameters
    ----------
    config
        Loader-specific configuration.
    """

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Yield dataset samples.

        Yields
        ------
        Sample
            Dataset samples consumed by the evaluation pipeline.
        """

        raise NotImplementedError


class Preprocessor(ABC):
    """Base class for sample-to-tensor preprocessing.

    Parameters
    ----------
    config
        Preprocessor-specific configuration.
    """

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        """Validate preprocessor configuration.

        Raises
        ------
        ValueError
            Raised by implementations when configuration is invalid.
        """

        return None

    @abstractmethod
    def __call__(self, sample: Sample) -> TensorDict:
        """Convert a sample into normalized model inputs.

        Parameters
        ----------
        sample
            Sample emitted by a dataset loader.

        Returns
        -------
        TensorDict
            Common input tensor dictionary before backend-specific mapping.
        """

        raise NotImplementedError


class Postprocessor(ABC):
    """Base class for backend output postprocessing.

    Parameters
    ----------
    config
        Postprocessor-specific configuration.
    """

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        """Validate postprocessor configuration.

        Raises
        ------
        ValueError
            Raised by implementations when configuration is invalid.
        """

        return None

    @abstractmethod
    def __call__(self, outputs: TensorDict, sample: Sample | None = None) -> ComparableDict:
        """Convert raw backend outputs into task-level comparable values.

        Parameters
        ----------
        outputs
            Normalized raw output tensor dictionary.
        sample
            Optional source sample for metadata-aware postprocessing.

        Returns
        -------
        ComparableDict
            Postprocessed result dictionary consumed by post metrics.
        """

        raise NotImplementedError


class ArtifactBuilder(ABC):
    """Base class for model artifact builders.

    Parameters
    ----------
    name
        Variant name.
    backend
        Backend identifier.
    precision
        Optional precision label.
    config
        Builder-specific configuration.
    """

    def __init__(
        self,
        name: str,
        backend: str,
        precision: str | None = None,
        config: ConfigDict | None = None,
    ) -> None:
        self.name = name
        self.backend = backend
        self.precision = precision
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        """Validate builder configuration.

        Raises
        ------
        ValueError
            Raised by implementations when configuration is invalid.
        """

        return None

    @abstractmethod
    def build(self, source_model: SourceModel) -> ModelArtifact:
        """Build or resolve a runtime artifact.

        Parameters
        ----------
        source_model
            Source model used to produce the backend artifact.

        Returns
        -------
        ModelArtifact
            Runtime artifact consumed by a runner.
        """

        raise NotImplementedError


class ModelRunner(ABC):
    """Base class for backend inference runners.

    Parameters
    ----------
    name
        Variant name.
    artifact
        Runtime artifact produced by the corresponding builder.
    config
        Runner-specific configuration.
    """

    def __init__(
        self,
        name: str,
        artifact: ModelArtifact,
        config: ConfigDict | None = None,
    ) -> None:
        self.name = name
        self.artifact = artifact
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        """Validate runner configuration.

        Raises
        ------
        ValueError
            Raised by implementations when configuration is invalid.
        """

        return None

    @abstractmethod
    def load(self) -> None:
        """Load backend runtime resources.

        Raises
        ------
        RuntimeError
            Raised when required backend libraries or runtime resources are
            unavailable.
        """

        raise NotImplementedError

    @abstractmethod
    def infer(self, inputs: TensorDict) -> TensorDict:
        """Run inference.

        Parameters
        ----------
        inputs
            Backend input tensor dictionary after input mapping.

        Returns
        -------
        TensorDict
            Backend output tensor dictionary before output mapping.
        """

        raise NotImplementedError

    @abstractmethod
    def get_input_specs(self) -> list[TensorSpec]:
        """Return runner input metadata.

        Returns
        -------
        list[TensorSpec]
            Input tensor metadata known after loading the runner.
        """

        raise NotImplementedError

    @abstractmethod
    def get_output_specs(self) -> list[TensorSpec]:
        """Return runner output metadata.

        Returns
        -------
        list[TensorSpec]
            Output tensor metadata known after loading the runner.
        """

        raise NotImplementedError

    def warmup(self, inputs: TensorDict, runs: int = 3) -> None:
        """Run warmup inference calls.

        Parameters
        ----------
        inputs
            Backend input tensor dictionary.
        runs
            Number of warmup iterations.
        """

        for _ in range(runs):
            self.infer(inputs)


class Metric(ABC):
    """Base class for streaming metrics.

    Parameters
    ----------
    name
        Metric instance name.
    config
        Metric-specific configuration.
    """

    def __init__(self, name: str, config: ConfigDict | None = None) -> None:
        self.name = name
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        """Validate metric configuration.

        Raises
        ------
        ValueError
            Raised by implementations when configuration is invalid.
        """

        return None

    @abstractmethod
    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update metric state with one sample.

        Parameters
        ----------
        reference
            Reference output dictionary.
        target
            Target output dictionary.
        sample
            Optional evaluated sample.

        Returns
        -------
        list[SampleMetricRecord]
            Per-sample records emitted for the update.
        """

        raise NotImplementedError

    @abstractmethod
    def compute(self) -> list[MetricSummaryRecord]:
        """Compute aggregate metric records.

        Returns
        -------
        list[MetricSummaryRecord]
            Aggregated records for the evaluated samples.
        """

        raise NotImplementedError


class Reporter(ABC):
    """Base class for report writers.

    Parameters
    ----------
    config
        Reporter-specific configuration.
    """

    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def write(self, report_data: ReportData) -> None:
        """Write report data.

        Parameters
        ----------
        report_data
            Report-ready data generated from an evaluation result.
        """

        raise NotImplementedError
