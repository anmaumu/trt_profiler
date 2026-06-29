from __future__ import annotations

import numpy as np

from trt_profiler.core.types import Sample
from trt_profiler.metrics.detection import (
    DetectionAccuracyMetric,
    DetectionConsistencyMetric,
    box_iou,
)


def test_box_iou_computes_pairwise_iou() -> None:
    iou = box_iou(
        np.array([[0.0, 0.0, 10.0, 10.0]]),
        np.array([[5.0, 5.0, 15.0, 15.0]]),
    )

    assert np.isclose(iou[0, 0], 25.0 / 175.0)


def test_detection_consistency_metric_matches_reference_and_target() -> None:
    metric = DetectionConsistencyMetric(name="det_consistency", config={"iou_threshold": 0.5})

    records = metric.update(
        {
            "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([1]),
        },
        {
            "boxes": np.array([[1.0, 1.0, 11.0, 11.0]]),
            "scores": np.array([0.8]),
            "labels": np.array([1]),
        },
        Sample(id="sample", data={}),
    )
    summary = metric.compute()

    assert any(record.stat == "matched_count" and record.value == 1.0 for record in records)
    assert any(record.stat == "class_match_rate" and record.value == 1.0 for record in summary)
    assert any(record.stat == "mean_confidence_abs_diff" for record in summary)


def test_detection_accuracy_metric_computes_simple_map_from_annotations() -> None:
    metric = DetectionAccuracyMetric(
        name="det_accuracy",
        config={"iou_thresholds": [0.5], "ground_truth_source": "annotations"},
    )
    sample = Sample(
        id="sample",
        data={},
        annotations=[{"box": [0.0, 0.0, 10.0, 10.0], "label": 1}],
    )

    records = metric.update(
        {},
        {
            "boxes": np.array([[0.0, 0.0, 10.0, 10.0]]),
            "scores": np.array([0.9]),
            "labels": np.array([1]),
        },
        sample,
    )
    summary = metric.compute()

    assert any(record.stat == "ap@0.5" and record.value == 1.0 for record in records)
    assert any(record.stat == "map" and record.value == 1.0 for record in summary)
