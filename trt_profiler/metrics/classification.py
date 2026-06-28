from __future__ import annotations

from typing import Any

import numpy as np

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class ClassificationConsistencyMetric(Metric):
    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._top1_matches: list[float] = []

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        probs_key = str(self.config.get("probs_key", "probs"))
        ref_top1 = int(np.argmax(np.asarray(reference[probs_key])))
        tgt_top1 = int(np.argmax(np.asarray(target[probs_key])))
        match = ref_top1 == tgt_top1
        self._top1_matches.append(1.0 if match else 0.0)
        return [
            SampleMetricRecord(
                sample_id=sample.id if sample is not None else "",
                comparison="",
                stage="",
                metric=self.name,
                output=probs_key,
                stat="top1_match",
                value=match,
                status="pass" if match else "fail",
            )
        ]

    def compute(self) -> list[MetricSummaryRecord]:
        value = float(np.mean(self._top1_matches)) if self._top1_matches else 0.0
        return [
            MetricSummaryRecord(
                comparison="",
                stage="",
                metric=self.name,
                output=str(self.config.get("probs_key", "probs")),
                stat="top1_match_rate",
                value=value,
            )
        ]


class ClassificationAccuracyMetric(Metric):
    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        return []

    def compute(self) -> list[MetricSummaryRecord]:
        return []
