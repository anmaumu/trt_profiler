from __future__ import annotations

import numpy as np

from trt_profiler.core.types import Preprocessor, Sample, TensorDict


class NpzPreprocessor(Preprocessor):
    def __call__(self, sample: Sample) -> TensorDict:
        if not isinstance(sample.data, dict):
            raise TypeError("NpzPreprocessor expects Sample.data to be a dict.")

        inputs_config = self.config.get("inputs")
        if inputs_config is None:
            return {key: np.asarray(value) for key, value in sample.data.items()}

        result: TensorDict = {}
        for item in inputs_config:
            name = str(item["name"])
            npz_key = str(item.get("npz_key", name))
            array = np.asarray(sample.data[npz_key])
            dtype = item.get("dtype")
            if dtype is not None:
                array = array.astype(str(dtype))
            result[name] = array
        return result
