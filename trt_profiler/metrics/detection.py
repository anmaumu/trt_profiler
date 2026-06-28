from __future__ import annotations

from typing import Any

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class DetectionConsistencyMetric(Metric):
    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        return []

    def compute(self) -> list[MetricSummaryRecord]:
        return []


class DetectionAccuracyMetric(Metric):
    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        return []

    def compute(self) -> list[MetricSummaryRecord]:
        return []
