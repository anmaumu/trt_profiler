"""Generic tensor difference metric."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


@dataclass
class _OutputStats:
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, stat: str, value: float) -> None:
        self.values[stat].append(value)


class TensorDiffMetric(Metric):
    """Compare tensors with absolute, relative, and similarity statistics.

    Config Keys
    -----------
    outputs : list[str], optional
        Output names to compare. Defaults to all reference outputs.
    atol : float, optional
        Absolute tolerance for ``allclose_rate``.
    rtol : float, optional
        Relative tolerance for ``allclose_rate``.
    relative_eps : float, optional
        Lower bound used in relative error denominator.
    percentiles : list[float], optional
        Absolute-error percentiles to record.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._stats: dict[str, _OutputStats] = defaultdict(_OutputStats)

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update tensor difference statistics for one sample.

        Parameters
        ----------
        reference
            Reference output tensors.
        target
            Target output tensors.
        sample
            Optional evaluated sample.

        Returns
        -------
        list[SampleMetricRecord]
            Per-sample tensor difference records.
        """

        outputs = [str(output) for output in self.config.get("outputs", reference.keys())]
        records: list[SampleMetricRecord] = []
        sample_id = sample.id if sample is not None else ""

        for output in outputs:
            ref = np.asarray(reference[output])
            tgt = np.asarray(target[output])
            if ref.shape != tgt.shape:
                records.append(
                    SampleMetricRecord(
                        sample_id=sample_id,
                        comparison="",
                        stage="",
                        metric=self.name,
                        output=output,
                        stat="shape_match",
                        value=False,
                        status="fail",
                    )
                )
                self._stats[output].add("shape_match_rate", 0.0)
                continue

            ref_f = ref.astype(np.float64, copy=False)
            tgt_f = tgt.astype(np.float64, copy=False)
            diff = tgt_f - ref_f
            abs_diff = np.abs(diff)
            denom = np.maximum(np.abs(ref_f), float(self.config.get("relative_eps", 1e-12)))
            rel_diff = abs_diff / denom

            values = {
                "shape_match": True,
                "max_abs_error": float(np.max(abs_diff)) if abs_diff.size else 0.0,
                "mean_abs_error": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
                "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
                "max_rel_error": float(np.max(rel_diff)) if rel_diff.size else 0.0,
                "mean_rel_error": float(np.mean(rel_diff)) if rel_diff.size else 0.0,
                "cosine_similarity": _cosine_similarity(ref_f, tgt_f),
                "allclose_rate": float(
                    np.mean(
                        np.isclose(
                            ref_f,
                            tgt_f,
                            atol=float(self.config.get("atol", 1e-5)),
                            rtol=float(self.config.get("rtol", 1e-5)),
                        )
                    )
                )
                if ref_f.size
                else 1.0,
                "nan_count": float(np.isnan(tgt_f).sum()),
                "inf_count": float(np.isinf(tgt_f).sum()),
            }

            percentiles = [float(item) for item in self.config.get("percentiles", [])]
            for percentile in percentiles:
                values[f"p{percentile:g}_abs_error"] = (
                    float(np.percentile(abs_diff, percentile)) if abs_diff.size else 0.0
                )

            for stat, value in values.items():
                if isinstance(value, bool):
                    self._stats[output].add(f"{stat}_rate", 1.0 if value else 0.0)
                else:
                    self._stats[output].add(stat, float(value))
                records.append(
                    SampleMetricRecord(
                        sample_id=sample_id,
                        comparison="",
                        stage="",
                        metric=self.name,
                        output=output,
                        stat=stat,
                        value=value,
                    )
                )

        return records

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute aggregate tensor difference statistics.

        Returns
        -------
        list[MetricSummaryRecord]
            Mean statistics and max values for error-like statistics.
        """

        records: list[MetricSummaryRecord] = []
        for output, stats in self._stats.items():
            for stat, values in stats.values.items():
                records.append(
                    MetricSummaryRecord(
                        comparison="",
                        stage="",
                        metric=self.name,
                        output=output,
                        stat=stat,
                        value=float(np.mean(values)) if values else 0.0,
                    )
                )
                if stat not in {"cosine_similarity", "allclose_rate", "shape_match_rate"}:
                    records.append(
                        MetricSummaryRecord(
                            comparison="",
                            stage="",
                            metric=self.name,
                            output=output,
                            stat=f"{stat}_max",
                            value=float(np.max(values)) if values else 0.0,
                        )
                    )
        return records


def _cosine_similarity(reference: np.ndarray, target: np.ndarray) -> float:
    ref_flat = reference.reshape(-1)
    tgt_flat = target.reshape(-1)
    if ref_flat.size == 0:
        return 1.0
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(tgt_flat))
    if denom == 0.0:
        return 1.0 if np.allclose(ref_flat, tgt_flat) else 0.0
    return float(np.dot(ref_flat, tgt_flat) / denom)
