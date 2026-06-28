"""Detection-oriented comparison metrics."""

from __future__ import annotations

from typing import Any

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class DetectionConsistencyMetric(Metric):
    """Placeholder for detection-result consistency metrics."""

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update detection consistency.

        Parameters
        ----------
        reference
            Reference detection result.
        target
            Target detection result.
        sample
            Optional evaluated sample.

        Returns
        -------
        list[SampleMetricRecord]
            Empty list until the metric is implemented.
        """

        return []

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute detection consistency summaries.

        Returns
        -------
        list[MetricSummaryRecord]
            Empty list until the metric is implemented.
        """

        return []


class DetectionAccuracyMetric(Metric):
    """Placeholder for annotation-based detection accuracy metrics."""

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update detection accuracy.

        Parameters
        ----------
        reference
            Reference detection result.
        target
            Target detection result.
        sample
            Optional sample containing annotations.

        Returns
        -------
        list[SampleMetricRecord]
            Empty list until the metric is implemented.
        """

        return []

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute detection accuracy summaries.

        Returns
        -------
        list[MetricSummaryRecord]
            Empty list until the metric is implemented.
        """

        return []
