"""Validation for readable evaluation configs."""

from __future__ import annotations

from trt_profiler.config.schema import EvaluationConfig


def validate_evaluation_config(config: EvaluationConfig) -> None:
    """Validate a parsed evaluation config.

    Parameters
    ----------
    config
        Parsed readable configuration.

    Raises
    ------
    ValueError
        If required fields are missing or inconsistent.
    """

    if not config.model.name:
        raise ValueError("model.name is required.")
    if not config.model.source_path or config.model.source_path == "None":
        raise ValueError("model.path or model.source_path is required.")
    if not config.input.path or config.input.path == "None":
        raise ValueError("input.path is required.")
    if not config.variants:
        raise ValueError("variants must contain at least one backend variant.")
