"""Detection-oriented comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


@dataclass(frozen=True)
class _Detections:
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class _Match:
    reference_index: int
    target_index: int
    iou: float


class DetectionConsistencyMetric(Metric):
    """Compare reference and target detection outputs.

    Config Keys
    -----------
    boxes_key : str, optional
        Detection boxes key. Defaults to ``"boxes"``.
    scores_key : str, optional
        Detection confidence key. Defaults to ``"scores"``.
    labels_key : str, optional
        Detection class labels key. Defaults to ``"labels"``.
    iou_threshold : float, optional
        IoU threshold used for reference-target matching. Defaults to ``0.5``.
    class_aware : bool, optional
        Require matching labels during assignment. Defaults to ``True``.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._stats: dict[str, list[float]] = {
            "reference_box_count": [],
            "target_box_count": [],
            "box_count_diff": [],
            "matched_count": [],
            "match_rate": [],
            "mean_iou": [],
            "class_match_rate": [],
            "mean_confidence_abs_diff": [],
            "unmatched_reference_count": [],
            "unmatched_target_count": [],
        }

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update detection consistency for one sample.

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
            Per-sample detection consistency records.
        """

        ref = _detections_from_mapping(reference, self.config)
        tgt = _detections_from_mapping(target, self.config)
        matches = _match_detections(
            ref,
            tgt,
            iou_threshold=float(self.config.get("iou_threshold", 0.5)),
            class_aware=bool(self.config.get("class_aware", True)),
        )

        matched_count = len(matches)
        class_matches = [
            float(ref.labels[match.reference_index] == tgt.labels[match.target_index])
            for match in matches
        ]
        confidence_diffs = [
            abs(float(tgt.scores[match.target_index]) - float(ref.scores[match.reference_index]))
            for match in matches
        ]
        values = {
            "reference_box_count": float(len(ref.boxes)),
            "target_box_count": float(len(tgt.boxes)),
            "box_count_diff": float(len(tgt.boxes) - len(ref.boxes)),
            "matched_count": float(matched_count),
            "match_rate": float(matched_count / len(ref.boxes)) if len(ref.boxes) else 1.0,
            "mean_iou": float(np.mean([match.iou for match in matches])) if matches else 0.0,
            "class_match_rate": float(np.mean(class_matches)) if class_matches else 0.0,
            "mean_confidence_abs_diff": float(np.mean(confidence_diffs))
            if confidence_diffs
            else 0.0,
            "unmatched_reference_count": float(len(ref.boxes) - matched_count),
            "unmatched_target_count": float(len(tgt.boxes) - matched_count),
        }
        return _record_values(self.name, self._stats, sample, "detections", values)

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute detection consistency summaries.

        Returns
        -------
        list[MetricSummaryRecord]
            Aggregated consistency records.
        """

        return _summary_records(self.name, "detections", self._stats)


class DetectionAccuracyMetric(Metric):
    """Compute simplified detection AP/mAP against annotations or reference.

    Config Keys
    -----------
    boxes_key, scores_key, labels_key : str, optional
        Detection output keys. Defaults are ``"boxes"``, ``"scores"``, and
        ``"labels"``.
    iou_thresholds : list[float], optional
        IoU thresholds used for AP. Defaults to ``[0.5]``.
    ground_truth_source : str, optional
        ``"annotations"`` uses ``sample.annotations``. ``"reference"`` treats
        the reference result as ground truth. Defaults to ``"annotations"``.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._thresholds = [float(item) for item in self.config.get("iou_thresholds", [0.5])]
        self._stats: dict[str, list[float]] = {
            "prediction_count": [],
            "ground_truth_count": [],
            "map": [],
        }
        for threshold in self._thresholds:
            self._stats[f"ap@{threshold:g}"] = []
            self._stats[f"recall@{threshold:g}"] = []
            self._stats[f"precision@{threshold:g}"] = []

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        """Update simplified AP/mAP for one sample.

        Parameters
        ----------
        reference
            Reference detection result. Used as ground truth when configured.
        target
            Target detection result.
        sample
            Optional sample containing annotations.

        Returns
        -------
        list[SampleMetricRecord]
            Per-sample AP, precision, and recall records.
        """

        predictions = _detections_from_mapping(target, self.config)
        ground_truth = _ground_truth(reference, sample, self.config)

        values: dict[str, float] = {
            "prediction_count": float(len(predictions.boxes)),
            "ground_truth_count": float(len(ground_truth.boxes)),
        }
        aps = []
        for threshold in self._thresholds:
            precision, recall, ap = _average_precision_at_iou(
                predictions,
                ground_truth,
                iou_threshold=threshold,
            )
            values[f"precision@{threshold:g}"] = precision
            values[f"recall@{threshold:g}"] = recall
            values[f"ap@{threshold:g}"] = ap
            aps.append(ap)
        values["map"] = float(np.mean(aps)) if aps else 0.0
        return _record_values(self.name, self._stats, sample, "detections", values)

    def compute(self) -> list[MetricSummaryRecord]:
        """Compute simplified detection AP/mAP summaries.

        Returns
        -------
        list[MetricSummaryRecord]
            Aggregated AP, precision, recall, and count records.
        """

        return _summary_records(self.name, "detections", self._stats)


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU for xyxy boxes.

    Parameters
    ----------
    boxes_a
        Array with shape ``(N, 4)``.
    boxes_b
        Array with shape ``(M, 4)``.

    Returns
    -------
    np.ndarray
        IoU matrix with shape ``(N, M)``.
    """

    a = _as_boxes(boxes_a)
    b = _as_boxes(boxes_b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = _box_area(a)[:, None]
    area_b = _box_area(b)[None, :]
    union = area_a + area_b - intersection
    result: np.ndarray = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    return result


def _detections_from_mapping(data: dict[str, Any], config: dict[str, Any]) -> _Detections:
    boxes_key = str(config.get("boxes_key", "boxes"))
    scores_key = str(config.get("scores_key", "scores"))
    labels_key = str(config.get("labels_key", "labels"))
    boxes = _as_boxes(data.get(boxes_key, []))
    scores = np.asarray(data.get(scores_key, np.ones(len(boxes))), dtype=np.float64).reshape(-1)
    labels = np.asarray(data.get(labels_key, np.zeros(len(boxes))), dtype=np.int64).reshape(-1)
    if len(scores) != len(boxes) or len(labels) != len(boxes):
        raise ValueError("Detection boxes, scores, and labels must have the same length.")
    return _Detections(boxes=boxes, scores=scores, labels=labels)


def _ground_truth(
    reference: dict[str, Any],
    sample: Sample | None,
    config: dict[str, Any],
) -> _Detections:
    source = str(config.get("ground_truth_source", "annotations"))
    if source == "reference":
        return _detections_from_mapping(reference, config)
    if source != "annotations":
        raise ValueError(f"Unsupported ground_truth_source: {source!r}")
    if sample is None:
        raise ValueError("DetectionAccuracyMetric requires sample annotations.")
    return _detections_from_annotations(sample.annotations)


def _detections_from_annotations(annotations: list[dict[str, Any]]) -> _Detections:
    boxes = []
    scores = []
    labels = []
    for annotation in annotations:
        box = annotation.get("box", annotation.get("bbox"))
        if box is None:
            continue
        boxes.append(box)
        scores.append(float(annotation.get("score", 1.0)))
        labels.append(int(annotation.get("label", annotation.get("class_id", 0))))
    return _Detections(
        boxes=_as_boxes(boxes),
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
    )


def _match_detections(
    reference: _Detections,
    target: _Detections,
    *,
    iou_threshold: float,
    class_aware: bool,
) -> list[_Match]:
    ious = box_iou(reference.boxes, target.boxes)
    candidates: list[tuple[float, int, int]] = []
    for ref_index in range(len(reference.boxes)):
        for target_index in range(len(target.boxes)):
            if ious[ref_index, target_index] < iou_threshold:
                continue
            if class_aware and reference.labels[ref_index] != target.labels[target_index]:
                continue
            candidates.append((float(ious[ref_index, target_index]), ref_index, target_index))

    matches: list[_Match] = []
    used_ref: set[int] = set()
    used_target: set[int] = set()
    for iou, ref_index, target_index in sorted(candidates, reverse=True):
        if ref_index in used_ref or target_index in used_target:
            continue
        matches.append(_Match(reference_index=ref_index, target_index=target_index, iou=iou))
        used_ref.add(ref_index)
        used_target.add(target_index)
    return matches


def _average_precision_at_iou(
    predictions: _Detections,
    ground_truth: _Detections,
    *,
    iou_threshold: float,
) -> tuple[float, float, float]:
    if len(ground_truth.boxes) == 0:
        return (1.0, 1.0, 1.0) if len(predictions.boxes) == 0 else (0.0, 0.0, 0.0)
    if len(predictions.boxes) == 0:
        return 0.0, 0.0, 0.0

    order = np.argsort(predictions.scores)[::-1]
    matched_gt: set[int] = set()
    true_positive = np.zeros(len(order), dtype=np.float64)
    false_positive = np.zeros(len(order), dtype=np.float64)
    ious = box_iou(predictions.boxes, ground_truth.boxes)

    for rank, pred_index in enumerate(order):
        same_class = np.where(ground_truth.labels == predictions.labels[pred_index])[0]
        best_gt = -1
        best_iou = 0.0
        for gt_index in same_class:
            iou = float(ious[pred_index, gt_index])
            if iou > best_iou:
                best_iou = iou
                best_gt = int(gt_index)
        if best_iou >= iou_threshold and best_gt not in matched_gt:
            true_positive[rank] = 1.0
            matched_gt.add(best_gt)
        else:
            false_positive[rank] = 1.0

    tp_cum = np.cumsum(true_positive)
    fp_cum = np.cumsum(false_positive)
    recalls = tp_cum / len(ground_truth.boxes)
    precisions = np.divide(
        tp_cum,
        tp_cum + fp_cum,
        out=np.zeros_like(tp_cum),
        where=(tp_cum + fp_cum) > 0.0,
    )
    precision = float(precisions[-1]) if precisions.size else 0.0
    recall = float(recalls[-1]) if recalls.size else 0.0
    return precision, recall, _interpolated_ap(recalls, precisions)


def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 11)
    values = []
    for point in recall_points:
        valid = precisions[recalls >= point]
        values.append(float(np.max(valid)) if valid.size else 0.0)
    return float(np.mean(values))


def _as_boxes(value: Any) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    boxes = boxes.reshape(-1, 4)
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    result: np.ndarray = np.stack([x1, y1, x2, y2], axis=1)
    return result


def _box_area(boxes: np.ndarray) -> np.ndarray:
    wh = np.clip(boxes[:, 2:] - boxes[:, :2], a_min=0.0, a_max=None)
    area: np.ndarray = wh[:, 0] * wh[:, 1]
    return area


def _record_values(
    metric_name: str,
    stats: dict[str, list[float]],
    sample: Sample | None,
    output: str,
    values: dict[str, float],
) -> list[SampleMetricRecord]:
    sample_id = sample.id if sample is not None else ""
    records = []
    for stat, value in values.items():
        stats[stat].append(float(value))
        records.append(
            SampleMetricRecord(
                sample_id=sample_id,
                comparison="",
                stage="",
                metric=metric_name,
                output=output,
                stat=stat,
                value=value,
            )
        )
    return records


def _summary_records(
    metric_name: str,
    output: str,
    stats: dict[str, list[float]],
) -> list[MetricSummaryRecord]:
    records: list[MetricSummaryRecord] = []
    for stat, values in stats.items():
        records.append(
            MetricSummaryRecord(
                comparison="",
                stage="",
                metric=metric_name,
                output=output,
                stat=stat,
                value=float(np.mean(values)) if values else 0.0,
            )
        )
        if stat.endswith("_diff") or stat.endswith("_count"):
            records.append(
                MetricSummaryRecord(
                    comparison="",
                    stage="",
                    metric=metric_name,
                    output=output,
                    stat=f"{stat}_max",
                    value=float(np.max(np.abs(values))) if values else 0.0,
                )
            )
    return records
