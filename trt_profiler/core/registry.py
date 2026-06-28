from __future__ import annotations

from trt_profiler.builders.onnx_passthrough import OnnxPassthroughBuilder
from trt_profiler.builders.openvino import OpenVinoBuilder
from trt_profiler.builders.tensorrt import TensorRTBuilder
from trt_profiler.data.image_loader import ImageFolderLoader
from trt_profiler.data.npz_loader import NpzFolderLoader
from trt_profiler.data.video_loader import VideoLoader
from trt_profiler.metrics.classification import (
    ClassificationAccuracyMetric,
    ClassificationConsistencyMetric,
)
from trt_profiler.metrics.detection import DetectionAccuracyMetric, DetectionConsistencyMetric
from trt_profiler.metrics.feature_map_diff import FeatureMapDiffMetric
from trt_profiler.metrics.tensor_diff import TensorDiffMetric
from trt_profiler.postprocessors.identity import IdentityPostprocessor
from trt_profiler.preprocessors.image import ImageNetPreprocessor
from trt_profiler.preprocessors.npz import NpzPreprocessor
from trt_profiler.report.json_reporter import JsonReporter
from trt_profiler.runners.identity import IdentityRunner
from trt_profiler.runners.onnxruntime import OnnxRuntimeRunner
from trt_profiler.runners.openvino import OpenVinoRunner
from trt_profiler.runners.tensorrt import TensorRTRunner

BUILTIN_CLASSES: dict[str, type[object]] = {
    "trt_profiler.builders.OnnxPassthroughBuilder": OnnxPassthroughBuilder,
    "trt_profiler.builders.OpenVinoBuilder": OpenVinoBuilder,
    "trt_profiler.builders.TensorRTBuilder": TensorRTBuilder,
    "trt_profiler.data.ImageFolderLoader": ImageFolderLoader,
    "trt_profiler.data.NpzFolderLoader": NpzFolderLoader,
    "trt_profiler.data.VideoLoader": VideoLoader,
    "trt_profiler.metrics.TensorDiffMetric": TensorDiffMetric,
    "trt_profiler.metrics.FeatureMapDiffMetric": FeatureMapDiffMetric,
    "trt_profiler.metrics.ClassificationConsistencyMetric": ClassificationConsistencyMetric,
    "trt_profiler.metrics.ClassificationAccuracyMetric": ClassificationAccuracyMetric,
    "trt_profiler.metrics.DetectionConsistencyMetric": DetectionConsistencyMetric,
    "trt_profiler.metrics.DetectionAccuracyMetric": DetectionAccuracyMetric,
    "trt_profiler.postprocessors.IdentityPostprocessor": IdentityPostprocessor,
    "trt_profiler.preprocessors.ImageNetPreprocessor": ImageNetPreprocessor,
    "trt_profiler.preprocessors.NpzPreprocessor": NpzPreprocessor,
    "trt_profiler.report.JsonReporter": JsonReporter,
    "trt_profiler.runners.IdentityRunner": IdentityRunner,
    "trt_profiler.runners.OnnxRuntimeRunner": OnnxRuntimeRunner,
    "trt_profiler.runners.OpenVinoRunner": OpenVinoRunner,
    "trt_profiler.runners.TensorRTRunner": TensorRTRunner,
}


DATASET_TYPES: dict[str, str] = {
    "image_folder": "trt_profiler.data.ImageFolderLoader",
    "npz_folder": "trt_profiler.data.NpzFolderLoader",
    "video": "trt_profiler.data.VideoLoader",
    "video_file": "trt_profiler.data.VideoLoader",
    "video_folder": "trt_profiler.data.VideoLoader",
}
