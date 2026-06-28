from __future__ import annotations

from trt_profiler.core.types import ModelRunner, TensorDict, TensorSpec


class TensorRTRunner(ModelRunner):
    """Placeholder native TensorRT runner.

    The class keeps the public contract in place. Engine loading and CUDA buffer
    management will be implemented behind this interface.
    """

    def load(self) -> None:
        if self.artifact.path is None:
            raise ValueError(f"TensorRT engine path is missing for {self.name}")
        if not self.artifact.path.exists():
            raise FileNotFoundError(self.artifact.path)

    def infer(self, inputs: TensorDict) -> TensorDict:
        raise NotImplementedError("Native TensorRT inference is not implemented yet.")

    def get_input_specs(self) -> list[TensorSpec]:
        return []

    def get_output_specs(self) -> list[TensorSpec]:
        return []
