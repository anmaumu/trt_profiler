"""Classification-oriented comparison metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class ClassificationConsistencyMetric(Metric):
    """Compare classification outputs between reference and target.

    Config Keys
    -----------
    probs_key : str, optional
        Key containing class scores, logits, or probabilities. Defaults to
        ``"probs"``.
    topk : list[int], optional
        Top-k values to evaluate. Defaults to ``[1]``.
    apply_softmax : bool, optional
        Apply softmax before distribution metrics. Defaults to ``False``.
    eps : float, optional
        Numerical epsilon for KL/JS divergence. Defaults to ``1e-12``.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._stats: dict[str, list[float]] = {
            "kl_divergence": [],
            "js_divergence": [],
            "mean_score_abs_diff": [],
            "max_score_abs_diff": [],
            "topk_ranking_match": [],
        }
        self._topk_values = [int(item) for item in self.config.get("topk", [1])]
        for topk in self._topk_values:
            self._stats[f"top{topk}_match"] = []

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update classification consistency for one sample.

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
            Per-sample classification consistency records.
        """

        probs_key = str(self.config.get("probs_key", "probs"))
        ref_scores = _as_1d_scores(reference[probs_key])
        tgt_scores = _as_1d_scores(target[probs_key])
        if ref_scores.shape != tgt_scores.shape:
            raise ValueError(
                f"Classification score shape mismatch: {ref_scores.shape} != {tgt_scores.shape}"
            )

        ref_probs = _as_distribution(ref_scores, bool(self.config.get("apply_softmax", False)))
        tgt_probs = _as_distribution(tgt_scores, bool(self.config.get("apply_softmax", False)))
        eps = float(self.config.get("eps", 1e-12))
        abs_diff = np.abs(tgt_probs - ref_probs)
        ref_order = _top_indices(ref_probs, ref_probs.size)
        tgt_order = _top_indices(tgt_probs, tgt_probs.size)

        values: dict[str, float | bool] = {
            "kl_divergence": _kl_divergence(ref_probs, tgt_probs, eps),
            "js_divergence": _js_divergence(ref_probs, tgt_probs, eps),
            "mean_score_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
            "max_score_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
            "topk_ranking_match": bool(np.array_equal(ref_order, tgt_order)),
        }
        for topk in self._topk_values:
            ref_topk = set(_top_indices(ref_probs, min(topk, ref_probs.size)).tolist())
            tgt_topk = set(_top_indices(tgt_probs, min(topk, tgt_probs.size)).tolist())
            values[f"top{topk}_match"] = ref_topk == tgt_topk

        sample_id = sample.id if sample is not None else ""
        records = []
        for stat, value in values.items():
            numeric_value = 1.0 if value is True else 0.0 if value is False else float(value)
            self._stats[stat].append(numeric_value)
            records.append(
                SampleMetricRecord(
                    sample_id=sample_id,
                    comparison="",
                    stage="",
                    metric=self.name,
                    output=probs_key,
                    stat=stat,
                    value=value,
                    status=_status_for_bool(value),
                )
            )
        return records

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute classification consistency summaries.

        Returns
        -------
        list[MetricSummaryRecord]
            Aggregated consistency records.
        """

        probs_key = str(self.config.get("probs_key", "probs"))
        records: list[MetricSummaryRecord] = []
        for stat, values in self._stats.items():
            summary_stat = f"{stat}_rate" if stat.endswith("_match") else stat
            records.append(
                MetricSummaryRecord(
                    comparison="",
                    stage="",
                    metric=self.name,
                    output=probs_key,
                    stat=summary_stat,
                    value=float(np.mean(values)) if values else 0.0,
                )
            )
            if stat.endswith("_diff") or stat.endswith("_divergence"):
                records.append(
                    MetricSummaryRecord(
                        comparison="",
                        stage="",
                        metric=self.name,
                        output=probs_key,
                        stat=f"{stat}_max",
                        value=float(np.max(values)) if values else 0.0,
                    )
                )
        return records


class ClassificationAccuracyMetric(Metric):
    """Compute label-based top-k classification accuracy.

    Config Keys
    -----------
    probs_key : str, optional
        Key containing class scores or probabilities. Defaults to ``"probs"``.
    topk : list[int], optional
        Top-k values to evaluate. Defaults to ``[1, 5]``.
    label_key : str, optional
        Key in ``sample.metadata`` used when ``sample.label`` is not set.
        Defaults to ``"label"``.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._topk_values = [int(item) for item in self.config.get("topk", [1, 5])]
        self._stats: dict[str, list[float]] = {
            f"top{topk}_accuracy": [] for topk in self._topk_values
        }

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update label-based accuracy for one sample.

        Parameters
        ----------
        reference
            Unused reference result dictionary.
        target
            Target result dictionary.
        sample
            Sample containing ``label`` or configured metadata label.

        Returns
        -------
        list[SampleMetricRecord]
            Per-sample top-k accuracy records.
        """

        del reference
        if sample is None:
            raise ValueError("ClassificationAccuracyMetric requires sample labels.")
        label = _sample_label(sample, str(self.config.get("label_key", "label")))
        scores = _as_1d_scores(target[str(self.config.get("probs_key", "probs"))])

        records = []
        for topk in self._topk_values:
            predictions = set(_top_indices(scores, min(topk, scores.size)).tolist())
            hit = int(label) in predictions
            stat = f"top{topk}_accuracy"
            self._stats[stat].append(1.0 if hit else 0.0)
            records.append(
                SampleMetricRecord(
                    sample_id=sample.id,
                    comparison="",
                    stage="",
                    metric=self.name,
                    output=str(self.config.get("probs_key", "probs")),
                    stat=stat,
                    value=hit,
                    status="pass" if hit else "fail",
                )
            )
        return records

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute label-based top-k accuracy.

        Returns
        -------
        list[MetricSummaryRecord]
            Aggregated top-k accuracy records.
        """

        output = str(self.config.get("probs_key", "probs"))
        return [
            MetricSummaryRecord(
                comparison="",
                stage="",
                metric=self.name,
                output=output,
                stat=stat,
                value=float(np.mean(values)) if values else 0.0,
            )
            for stat, values in self._stats.items()
        ]


def _as_1d_scores(value: Any) -> np.ndarray:
    array: np.ndarray = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Classification scores must not be empty.")
    return array


def _as_distribution(scores: np.ndarray, apply_softmax: bool) -> np.ndarray:
    if apply_softmax:
        shifted = scores - np.max(scores)
        exp = np.exp(shifted)
        result: np.ndarray = exp / np.sum(exp)
        return result
    total = float(np.sum(scores))
    if total > 0.0 and np.all(scores >= 0.0):
        normalized: np.ndarray = scores / total
        return normalized
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    result = exp / np.sum(exp)
    return result


def _top_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    result: np.ndarray = np.argsort(scores)[::-1][:topk]
    return result


def _kl_divergence(reference: np.ndarray, target: np.ndarray, eps: float) -> float:
    ref = np.clip(reference, eps, 1.0)
    tgt = np.clip(target, eps, 1.0)
    ref = ref / np.sum(ref)
    tgt = tgt / np.sum(tgt)
    return float(np.sum(ref * np.log(ref / tgt)))


def _js_divergence(reference: np.ndarray, target: np.ndarray, eps: float) -> float:
    midpoint = 0.5 * (reference + target)
    return 0.5 * _kl_divergence(reference, midpoint, eps) + 0.5 * _kl_divergence(
        target, midpoint, eps
    )


def _status_for_bool(value: float | bool) -> str | None:
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return None


def _sample_label(sample: Sample, label_key: str) -> int:
    if sample.label is not None:
        return int(sample.label)
    if label_key in sample.metadata:
        return int(sample.metadata[label_key])
    raise ValueError(f"Sample {sample.id!r} does not contain a classification label.")
