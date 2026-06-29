from __future__ import annotations

from pathlib import Path

import numpy as np

from trt_profiler.core.types import Sample
from trt_profiler.metrics.feature_map_diff import FeatureMapDiffMetric


def test_feature_map_diff_metric_computes_channelwise_and_worst_sample(tmp_path: Path) -> None:
    metric = FeatureMapDiffMetric(
        name="feature_diff",
        config={
            "outputs": ["feat"],
            "channel_axis": 1,
            "save_heatmaps": True,
            "heatmap_dir": str(tmp_path),
        },
    )
    reference = np.ones((1, 2, 3, 3), dtype=np.float32)
    target = reference.copy()
    target[:, 1, :, :] += 0.5

    records = metric.update(
        {"feat": reference},
        {"feat": target},
        Sample(id="sample:1", data={}),
    )
    summary = metric.compute()

    assert any(record.stat == "layer_cosine_similarity" for record in records)
    assert any(record.stat == "heatmap_path" for record in records)
    assert any(
        record.stat == "worst_sample_id" and record.value == "sample:1" for record in summary
    )
    assert list(tmp_path.glob("*.npy"))
