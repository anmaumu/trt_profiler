from __future__ import annotations

import numpy as np

from trt_profiler.core.types import Sample
from trt_profiler.metrics.classification import (
    ClassificationAccuracyMetric,
    ClassificationConsistencyMetric,
)


def test_classification_consistency_metric_computes_topk_and_divergence() -> None:
    metric = ClassificationConsistencyMetric(
        name="cls_consistency",
        config={"probs_key": "probs", "topk": [1, 2]},
    )

    records = metric.update(
        {"probs": np.array([0.1, 0.7, 0.2])},
        {"probs": np.array([0.2, 0.6, 0.2])},
        Sample(id="sample", data={}),
    )
    summary = metric.compute()

    assert any(record.stat == "top1_match" and record.value is True for record in records)
    assert any(record.stat == "kl_divergence" for record in records)
    assert any(record.stat == "js_divergence" for record in records)
    assert any(record.stat == "top1_match_rate" and record.value == 1.0 for record in summary)


def test_classification_accuracy_metric_uses_sample_label() -> None:
    metric = ClassificationAccuracyMetric(
        name="cls_accuracy",
        config={"probs_key": "probs", "topk": [1, 2]},
    )

    records = metric.update(
        {},
        {"probs": np.array([0.1, 0.2, 0.7])},
        Sample(id="sample", data={}, label=2),
    )
    summary = metric.compute()

    assert all(record.value is True for record in records)
    assert any(record.stat == "top1_accuracy" and record.value == 1.0 for record in summary)
