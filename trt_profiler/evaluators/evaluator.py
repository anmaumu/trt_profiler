"""Metric evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class Evaluator:
    """Apply a set of metrics for one evaluation stage.

    Parameters
    ----------
    stage
        Stage name such as ``"raw"`` or ``"post"``.
    metrics
        Metrics evaluated for the stage.
    """

    def __init__(self, stage: str, metrics: list[Metric]) -> None:
        self.stage = stage
        self.metrics = metrics
        self._records: list[SampleMetricRecord] = []

    @property
    def records(self) -> list[SampleMetricRecord]:
        """Return accumulated per-sample records.

        Returns
        -------
        list[SampleMetricRecord]
            Records emitted by metric updates.
        """

        return self._records

    def update(
        self,
        comparison_name: str,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> None:
        """Update all metrics with one sample comparison.

        Parameters
        ----------
        comparison_name
            Name of the active comparison.
        reference
            Reference result dictionary.
        target
            Target result dictionary.
        sample
            Optional evaluated sample.
        """

        for metric in self.metrics:
            records = metric.update(reference, target, sample)
            for record in records:
                self._records.append(
                    SampleMetricRecord(
                        sample_id=record.sample_id,
                        comparison=comparison_name,
                        stage=self.stage,
                        metric=metric.name,
                        stat=record.stat,
                        value=record.value,
                        output=record.output,
                        threshold=record.threshold,
                        status=record.status,
                        source_path=record.source_path
                        or (str(sample.source_path) if sample and sample.source_path else None),
                    )
                )

    def compute(self, comparison_name: str) -> list[MetricSummaryRecord]:
        """Compute summaries for all metrics.

        Parameters
        ----------
        comparison_name
            Name of the active comparison.

        Returns
        -------
        list[MetricSummaryRecord]
            Summary records with comparison and stage filled in.
        """

        summaries: list[MetricSummaryRecord] = []
        for metric in self.metrics:
            for record in metric.compute():
                summaries.append(
                    MetricSummaryRecord(
                        comparison=comparison_name,
                        stage=self.stage,
                        metric=metric.name,
                        stat=record.stat,
                        value=record.value,
                        output=record.output,
                        threshold=record.threshold,
                        status=record.status,
                    )
                )
        return summaries


def nested_summary(records: list[MetricSummaryRecord]) -> dict[str, Any]:
    """Convert flat summary records into nested report structure.

    Parameters
    ----------
    records
        Flat metric summary records.

    Returns
    -------
    dict[str, Any]
        Nested summary keyed by comparison, stage, metric, output, and stat.
    """

    summary: dict[str, Any] = {"comparisons": defaultdict(lambda: defaultdict(dict))}
    comparisons = summary["comparisons"]
    for record in records:
        metric_bucket = comparisons[record.comparison][record.stage].setdefault(record.metric, {})
        output_key = record.output or "_"
        output_bucket = metric_bucket.setdefault(output_key, {})
        output_bucket[record.stat] = record.value
    summary["comparisons"] = {
        comparison: dict(stages) for comparison, stages in comparisons.items()
    }
    return summary
