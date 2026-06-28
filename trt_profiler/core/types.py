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
    name: str
    shape: tuple[int | str | None, ...]
    dtype: str


@dataclass(frozen=True)
class SourceModel:
    path: Path
    format: str = "onnx"


@dataclass(frozen=True)
class ModelArtifact:
    variant_name: str
    backend: str
    precision: str | None
    path: Path | None
    config: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class BackendVariant:
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
    name: str
    reference: str
    target: str


@dataclass(frozen=True)
class Sample:
    id: str
    data: Any
    source_path: Path | None = None
    label: Any | None = None
    annotations: list[dict[str, Any]] = field(default_factory=list)
    metadata: ConfigDict = field(default_factory=dict)


@dataclass(frozen=True)
class SampleMetricRecord:
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
    metadata: ConfigDict
    summary: ConfigDict
    per_sample: list[SampleMetricRecord] = field(default_factory=list)


@dataclass
class ReportData:
    metadata: ConfigDict
    summary: ConfigDict
    tables: dict[str, list[dict[str, Any]]]
    artifacts: ConfigDict = field(default_factory=dict)


class DatasetLoader(ABC):
    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        raise NotImplementedError


class Preprocessor(ABC):
    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        return None

    @abstractmethod
    def __call__(self, sample: Sample) -> TensorDict:
        raise NotImplementedError


class Postprocessor(ABC):
    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        return None

    @abstractmethod
    def __call__(self, outputs: TensorDict, sample: Sample | None = None) -> ComparableDict:
        raise NotImplementedError


class ArtifactBuilder(ABC):
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
        return None

    @abstractmethod
    def build(self, source_model: SourceModel) -> ModelArtifact:
        raise NotImplementedError


class ModelRunner(ABC):
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
        return None

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(self, inputs: TensorDict) -> TensorDict:
        raise NotImplementedError

    @abstractmethod
    def get_input_specs(self) -> list[TensorSpec]:
        raise NotImplementedError

    @abstractmethod
    def get_output_specs(self) -> list[TensorSpec]:
        raise NotImplementedError

    def warmup(self, inputs: TensorDict, runs: int = 3) -> None:
        for _ in range(runs):
            self.infer(inputs)


class Metric(ABC):
    def __init__(self, name: str, config: ConfigDict | None = None) -> None:
        self.name = name
        self.config = config or {}
        self.validate_config()

    def validate_config(self) -> None:
        return None

    @abstractmethod
    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> list[MetricSummaryRecord]:
        raise NotImplementedError


class Reporter(ABC):
    def __init__(self, config: ConfigDict | None = None) -> None:
        self.config = config or {}

    @abstractmethod
    def write(self, report_data: ReportData) -> None:
        raise NotImplementedError
