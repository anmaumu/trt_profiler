from __future__ import annotations

from trt_profiler.core.types import TensorDict


class OutputMapper:
    def __init__(self, mapping: dict[str, dict[str, str]] | None = None) -> None:
        self.mapping = mapping or {}

    def map(self, runner_name: str, outputs: TensorDict) -> TensorDict:
        runner_mapping = self.mapping.get(runner_name)
        if not runner_mapping:
            return dict(outputs)

        mapped: TensorDict = {}
        for common_output_name, runner_output_name in runner_mapping.items():
            if runner_output_name not in outputs:
                raise KeyError(
                    f"Missing runner output {runner_output_name!r} for runner {runner_name!r}"
                )
            mapped[common_output_name] = outputs[runner_output_name]
        return mapped
