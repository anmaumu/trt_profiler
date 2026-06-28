from __future__ import annotations

from trt_profiler.metrics.tensor_diff import TensorDiffMetric


class FeatureMapDiffMetric(TensorDiffMetric):
    """Feature map comparison metric.

    MVP behavior reuses tensor diff statistics. Feature-specific summaries such
    as channelwise cosine and spatial heatmaps can be added behind this class.
    """
