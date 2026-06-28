from __future__ import annotations

from collections import defaultdict
from typing import Any

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class Evaluator:
    def __init__(self, stage: str, metrics: list[Metric]) -> None:
        self.stage = stage
        self.metrics = metrics
        self._records: list[SampleMetricRecord] = []

    @property
    def records(self) -> list[SampleMetricRecord]:
        return self._records

    def update(
        self,
        comparison_name: str,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> None:
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
                    )
                )

    def compute(self, comparison_name: str) -> list[MetricSummaryRecord]:
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
