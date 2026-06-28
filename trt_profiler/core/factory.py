from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from trt_profiler.core.class_loader import load_class
from trt_profiler.core.registry import BUILTIN_CLASSES, DATASET_TYPES
from trt_profiler.core.types import (
    ArtifactBuilder,
    BackendVariant,
    ConfigDict,
    DatasetLoader,
    Metric,
    ModelArtifact,
    ModelRunner,
    Postprocessor,
    Preprocessor,
    Reporter,
)


def resolve_class(class_path: str) -> type[Any]:
    if class_path in BUILTIN_CLASSES:
        return BUILTIN_CLASSES[class_path]
    return load_class(class_path)


def build_dataset(config: ConfigDict) -> DatasetLoader:
    dataset_type = str(config["type"])
    class_path = str(config.get("class") or DATASET_TYPES[dataset_type])
    cls = resolve_class(class_path)
    return cast(DatasetLoader, cls(config=config.get("config", {})))


def build_preprocessor(config: ConfigDict) -> Preprocessor:
    cls = resolve_class(str(config["class"]))
    return cast(Preprocessor, cls(config=config.get("config", {})))


def build_postprocessors(configs: list[ConfigDict]) -> list[Postprocessor]:
    postprocessors: list[Postprocessor] = []
    for config in configs:
        cls = resolve_class(str(config["class"]))
        postprocessors.append(cast(Postprocessor, cls(config=config.get("config", {}))))
    return postprocessors


def build_metric(config: ConfigDict) -> Metric:
    cls = resolve_class(str(config["class"]))
    return cast(Metric, cls(name=str(config["name"]), config=config.get("config", {})))


def build_reporter(config: ConfigDict) -> Reporter:
    cls = resolve_class(str(config["class"]))
    return cast(Reporter, cls(config=config.get("config", {})))


def parse_variants(configs: list[ConfigDict]) -> list[BackendVariant]:
    variants: list[BackendVariant] = []
    for config in configs:
        builder = config["builder"]
        runner = config["runner"]
        variants.append(
            BackendVariant(
                name=str(config["name"]),
                backend=str(config["backend"]),
                role=str(config.get("role", "target")),
                precision=cast(str | None, config.get("precision")),
                builder_class=str(builder["class"]),
                runner_class=str(runner["class"]),
                builder_config=cast(ConfigDict, builder.get("config", {})),
                runner_config=cast(ConfigDict, runner.get("config", {})),
            )
        )
    return variants


def build_artifact_builder(variant: BackendVariant) -> ArtifactBuilder:
    cls = resolve_class(variant.builder_class)
    return cast(
        ArtifactBuilder,
        cls(
            name=variant.name,
            backend=variant.backend,
            precision=variant.precision,
            config=variant.builder_config,
        ),
    )


def build_runner(variant: BackendVariant, artifact: ModelArtifact) -> ModelRunner:
    cls = resolve_class(variant.runner_class)
    return cast(
        ModelRunner,
        cls(name=variant.name, artifact=artifact, config=variant.runner_config),
    )


def coerce_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)
