from __future__ import annotations

from typing import Any

from trt_profiler.core.types import ModelRunner, TensorDict, TensorSpec


class OpenVinoRunner(ModelRunner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compiled_model: Any | None = None

    def load(self) -> None:
        try:
            from openvino import Core
        except ImportError as exc:
            raise RuntimeError(
                "openvino is not installed. Install trt-profiler[openvino]."
            ) from exc

        if self.artifact.path is None:
            raise ValueError(f"OpenVINO artifact path is missing for {self.name}")
        core = Core()
        model = core.read_model(str(self.artifact.path))
        self._compiled_model = core.compile_model(model, self.config.get("device", "CPU"))

    def infer(self, inputs: TensorDict) -> TensorDict:
        if self._compiled_model is None:
            raise RuntimeError(f"Runner is not loaded: {self.name}")
        result = self._compiled_model(inputs)
        outputs = getattr(self._compiled_model, "outputs", [])
        mapped: TensorDict = {}
        for output in outputs:
            name = output.get_any_name()
            mapped[str(name)] = result[output]
        if mapped:
            return mapped
        return {str(key): value for key, value in result.items()}

    def get_input_specs(self) -> list[TensorSpec]:
        return []

    def get_output_specs(self) -> list[TensorSpec]:
        return []
