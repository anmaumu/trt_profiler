from __future__ import annotations

from trt_profiler.core.types import TensorDict


class InputMapper:
    def __init__(self, mapping: dict[str, dict[str, str]] | None = None) -> None:
        self.mapping = mapping or {}

    def map(self, runner_name: str, inputs: TensorDict) -> TensorDict:
        runner_mapping = self.mapping.get(runner_name)
        if not runner_mapping:
            return dict(inputs)

        mapped: TensorDict = {}
        for runner_input_name, common_input_name in runner_mapping.items():
            if common_input_name not in inputs:
                raise KeyError(
                    f"Missing common input {common_input_name!r} for runner {runner_name!r}"
                )
            mapped[runner_input_name] = inputs[common_input_name]
        return mapped
