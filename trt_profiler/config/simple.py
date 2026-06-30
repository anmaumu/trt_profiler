"""Compatibility helpers for concise config normalization."""

from __future__ import annotations

from trt_profiler.config.builder import build_pipeline_config
from trt_profiler.config.parser import parse_evaluation_config
from trt_profiler.config.validation import validate_evaluation_config
from trt_profiler.core.types import ConfigDict


def normalize_config(config: ConfigDict) -> ConfigDict:
    """Normalize a full or concise config into the full pipeline schema.

    This function is kept for callers that already import
    ``trt_profiler.config.simple.normalize_config``. New code should prefer the
    explicit loader flow in ``trt_profiler.config.loader``.

    Parameters
    ----------
    config
        Raw loaded config.

    Returns
    -------
    ConfigDict
        Full pipeline config.
    """

    if "common" in config:
        return config

    parsed = parse_evaluation_config(config)
    validate_evaluation_config(parsed)
    return build_pipeline_config(parsed)
