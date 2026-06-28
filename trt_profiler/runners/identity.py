from __future__ import annotations

import numpy as np

from trt_profiler.core.types import ModelRunner, TensorDict, TensorSpec


class IdentityRunner(ModelRunner):
    """Testing runner that returns selected inputs as outputs."""

    def load(self) -> None:
        pass

    def infer(self, inputs: TensorDict) -> TensorDict:
        output_mapping = self.config.get("outputs")
        scale = float(self.config.get("scale", 1.0))
        bias = float(self.config.get("bias", 0.0))

        if output_mapping is None:
            return {key: np.asarray(value) * scale + bias for key, value in inputs.items()}

        return {
            str(output_name): np.asarray(inputs[str(input_name)]) * scale + bias
            for output_name, input_name in output_mapping.items()
        }

    def get_input_specs(self) -> list[TensorSpec]:
        return []

    def get_output_specs(self) -> list[TensorSpec]:
        return []
