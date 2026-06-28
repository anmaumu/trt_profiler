from __future__ import annotations

from trt_profiler.core.types import ComparableDict, Postprocessor, Sample, TensorDict


class IdentityPostprocessor(Postprocessor):
    def __call__(self, outputs: TensorDict, sample: Sample | None = None) -> ComparableDict:
        return {key: value for key, value in outputs.items()}
