"""Classification-oriented comparison metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class ClassificationConsistencyMetric(Metric):
    """Compare top-1 predictions between reference and target outputs.

    Config Keys
    -----------
    probs_key : str, optional
        Key containing class scores or probabilities. Defaults to ``"probs"``.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._top1_matches: list[float] = []

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update top-1 consistency for one sample.

        Parameters
        ----------
        reference
            Reference postprocessed result.
        target
            Target postprocessed result.
        sample
            Optional evaluated sample.

        Returns
        -------
        list[SampleMetricRecord]
            Per-sample top-1 match record.
        """

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
        """Compute top-1 match rate.

        Returns
        -------
        list[MetricSummaryRecord]
            Aggregated top-1 match rate.
        """

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
    """Placeholder for label-based classification accuracy."""

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update label-based accuracy.

        Parameters
        ----------
        reference
            Reference result dictionary.
        target
            Target result dictionary.
        sample
            Optional sample with label metadata.

        Returns
        -------
        list[SampleMetricRecord]
            Empty list until the metric is implemented.
        """

        return []

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute label-based accuracy.

        Returns
        -------
        list[MetricSummaryRecord]
            Empty list until the metric is implemented.
        """

        return []
