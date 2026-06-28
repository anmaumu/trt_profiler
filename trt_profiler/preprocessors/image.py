from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from trt_profiler.core.types import Preprocessor, Sample, TensorDict


class ImageNetPreprocessor(Preprocessor):
    """Simple ImageNet-style preprocessor for RGB image files."""

    def __call__(self, sample: Sample) -> TensorDict:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("pillow is required for ImageNetPreprocessor.") from exc

        image = _load_image(sample, Image)
        size = self.config.get("size", [224, 224])
        width, height = int(size[0]), int(size[1])
        image = image.resize((width, height))

        array = np.asarray(image).astype(np.float32) / 255.0
        mean = np.asarray(self.config.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.asarray(self.config.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        array = (array - mean) / std

        layout = str(self.config.get("layout", "NCHW")).upper()
        if layout == "NCHW":
            array = np.transpose(array, (2, 0, 1))
        elif layout != "NHWC":
            raise ValueError(f"Unsupported layout: {layout}")

        array = np.expand_dims(array, axis=0)
        input_name = str(self.config.get("input_name", "input"))
        return {input_name: array.astype(str(self.config.get("dtype", "float32")))}


def _sample_path(sample: Sample) -> Path:
    if isinstance(sample.data, (str, Path)):
        return Path(sample.data)
    if sample.source_path is not None:
        return sample.source_path
    raise ValueError("Image sample must provide a path in data or source_path.")


def _load_image(sample: Sample, image_cls: Any) -> Any:
    if isinstance(sample.data, np.ndarray):
        array = sample.data
        color_format = str(sample.metadata.get("color_format", "RGB")).upper()
        if color_format == "BGR":
            array = array[..., ::-1]
        return image_cls.fromarray(array.astype(np.uint8), mode="RGB")

    image_path = _sample_path(sample)
    return image_cls.open(image_path).convert("RGB")
