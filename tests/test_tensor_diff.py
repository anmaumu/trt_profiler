from __future__ import annotations

import numpy as np

from trt_profiler.core.types import Sample
from trt_profiler.metrics.tensor_diff import TensorDiffMetric


def test_tensor_diff_metric_computes_summary() -> None:
    metric = TensorDiffMetric(
        name="tensor_diff",
        config={"outputs": ["logits"], "percentiles": [95]},
    )

    records = metric.update(
        {"logits": np.array([1.0, 2.0], dtype=np.float32)},
        {"logits": np.array([1.0, 2.1], dtype=np.float32)},
        Sample(id="sample", data={}),
    )
    summary = metric.compute()

    assert records
    assert any(record.stat == "mean_abs_error" for record in records)
    assert any(record.stat == "mean_abs_error" for record in summary)
