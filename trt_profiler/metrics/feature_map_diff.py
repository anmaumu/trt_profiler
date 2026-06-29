"""Feature map tensor comparison metric."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from trt_profiler.core.types import MetricSummaryRecord, Sample, SampleMetricRecord
from trt_profiler.metrics.tensor_diff import TensorDiffMetric


class FeatureMapDiffMetric(TensorDiffMetric):
    """Compare feature maps with layer, channel, and spatial diagnostics.

    Config Keys
    -----------
    outputs : list[str], optional
        Feature layer names to compare.
    channel_axis : int, optional
        Channel axis used for channelwise cosine. Defaults to ``1`` for NCHW.
    save_heatmaps : bool, optional
        Save per-sample spatial absolute-error heatmaps as ``.npy`` files.
        Defaults to ``False``.
    heatmap_dir : str, optional
        Directory for saved heatmaps. Defaults to ``"artifacts/heatmaps"``.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._feature_stats: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._worst_sample_by_layer: dict[str, tuple[str, float]] = {}

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update feature map statistics for one sample.

        Parameters
        ----------
        reference
            Reference feature tensors.
        target
            Target feature tensors.
        sample
            Optional evaluated sample.

        Returns
        -------
        list[SampleMetricRecord]
            Tensor diff records plus feature-specific records.
        """

        records = super().update(reference, target, sample)
        outputs = [str(output) for output in self.config.get("outputs", reference.keys())]
        sample_id = sample.id if sample is not None else ""

        for output in outputs:
            ref = np.asarray(reference[output], dtype=np.float64)
            tgt = np.asarray(target[output], dtype=np.float64)
            if ref.shape != tgt.shape:
                continue
            diff = tgt - ref
            abs_diff = np.abs(diff)
            layer_cosine = _cosine(ref, tgt)
            channel_cosines = _channelwise_cosine(
                ref,
                tgt,
                channel_axis=int(self.config.get("channel_axis", 1)),
            )
            mean_channel_cosine = (
                float(np.mean(channel_cosines)) if channel_cosines.size else layer_cosine
            )
            min_channel_cosine = (
                float(np.min(channel_cosines)) if channel_cosines.size else layer_cosine
            )
            spatial_mean_abs_error = _spatial_mean_abs_error(abs_diff)
            worst_error = float(np.max(abs_diff)) if abs_diff.size else 0.0
            heatmap_path = self._save_heatmap(output, sample_id, spatial_mean_abs_error)

            values: dict[str, float | str] = {
                "layer_cosine_similarity": layer_cosine,
                "mean_channel_cosine_similarity": mean_channel_cosine,
                "min_channel_cosine_similarity": min_channel_cosine,
                "spatial_mean_abs_error": float(np.mean(spatial_mean_abs_error))
                if spatial_mean_abs_error.size
                else 0.0,
                "worst_abs_error": worst_error,
            }
            if heatmap_path is not None:
                values["heatmap_path"] = heatmap_path

            self._update_worst_sample(output, sample_id, worst_error)
            for stat, value in values.items():
                if isinstance(value, str):
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
                    continue
                self._feature_stats[output][stat].append(float(value))
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
        """Compute feature map summary records.

        Returns
        -------
        list[MetricSummaryRecord]
            Tensor diff summaries plus feature-specific summaries and worst
            sample records.
        """

        records = super().compute()
        for output, stats in self._feature_stats.items():
            for stat, values in stats.items():
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
                records.append(
                    MetricSummaryRecord(
                        comparison="",
                        stage="",
                        metric=self.name,
                        output=output,
                        stat=f"{stat}_worst",
                        value=_worst_value(stat, values),
                    )
                )
            worst_sample = self._worst_sample_by_layer.get(output)
            if worst_sample is not None:
                records.append(
                    MetricSummaryRecord(
                        comparison="",
                        stage="",
                        metric=self.name,
                        output=output,
                        stat="worst_sample_id",
                        value=worst_sample[0],
                    )
                )
                records.append(
                    MetricSummaryRecord(
                        comparison="",
                        stage="",
                        metric=self.name,
                        output=output,
                        stat="worst_sample_abs_error",
                        value=worst_sample[1],
                    )
                )
        return records

    def _save_heatmap(
        self,
        output: str,
        sample_id: str,
        heatmap: np.ndarray,
    ) -> str | None:
        if not bool(self.config.get("save_heatmaps", False)):
            return None
        heatmap_dir = Path(str(self.config.get("heatmap_dir", "artifacts/heatmaps")))
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        safe_sample_id = sample_id.replace("/", "_").replace(":", "_") or "sample"
        path = heatmap_dir / f"{safe_sample_id}_{output}.npy"
        np.save(path, heatmap)
        return str(path)

    def _update_worst_sample(self, output: str, sample_id: str, value: float) -> None:
        current = self._worst_sample_by_layer.get(output)
        if current is None or value > current[1]:
            self._worst_sample_by_layer[output] = (sample_id, value)


def _cosine(reference: np.ndarray, target: np.ndarray) -> float:
    ref_flat = reference.reshape(-1)
    tgt_flat = target.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(tgt_flat))
    if denom == 0.0:
        return 1.0 if np.allclose(ref_flat, tgt_flat) else 0.0
    return float(np.dot(ref_flat, tgt_flat) / denom)


def _channelwise_cosine(
    reference: np.ndarray,
    target: np.ndarray,
    *,
    channel_axis: int,
) -> np.ndarray:
    if reference.ndim <= abs(channel_axis):
        empty: np.ndarray = np.asarray([], dtype=np.float64)
        return empty
    axis = channel_axis if channel_axis >= 0 else reference.ndim + channel_axis
    ref = np.moveaxis(reference, axis, 0).reshape(reference.shape[axis], -1)
    tgt = np.moveaxis(target, axis, 0).reshape(target.shape[axis], -1)
    values = []
    for ref_channel, tgt_channel in zip(ref, tgt, strict=True):
        values.append(_cosine(ref_channel, tgt_channel))
    result: np.ndarray = np.asarray(values, dtype=np.float64)
    return result


def _spatial_mean_abs_error(abs_diff: np.ndarray) -> np.ndarray:
    if abs_diff.ndim < 2:
        reshaped: np.ndarray = abs_diff.reshape(1, -1)
        return reshaped
    if abs_diff.ndim == 4:
        mean4: np.ndarray = np.mean(abs_diff, axis=(0, 1))
        return mean4
    if abs_diff.ndim == 3:
        mean3: np.ndarray = np.mean(abs_diff, axis=0)
        return mean3
    return abs_diff


def _worst_value(stat: str, values: list[float]) -> float:
    if not values:
        return 0.0
    if "cosine" in stat:
        return float(np.min(values))
    return float(np.max(values))
