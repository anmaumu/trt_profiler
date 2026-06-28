"""OpenVINO inference runner."""

from __future__ import annotations

from typing import Any

from trt_profiler.core.types import ModelRunner, TensorDict, TensorSpec


class OpenVinoRunner(ModelRunner):
    """Run inference with OpenVINO Runtime.

    Config Keys
    -----------
    device : str, optional
        OpenVINO device name. Defaults to ``"CPU"``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compiled_model: Any | None = None

    def load(self) -> None:
        """Compile the model with OpenVINO.

        Raises
        ------
        RuntimeError
            If OpenVINO is not installed.
        ValueError
            If the artifact path is missing.
        """

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
        """Run OpenVINO inference.

        Parameters
        ----------
        inputs
            Backend input tensor dictionary.

        Returns
        -------
        TensorDict
            Output tensors keyed by OpenVINO output names.
        """

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
        """Return input metadata.

        Returns
        -------
        list[TensorSpec]
            Empty list for the current minimal OpenVINO runner.
        """

        return []

    def get_output_specs(self) -> list[TensorSpec]:
        """Return output metadata.

        Returns
        -------
        list[TensorSpec]
            Empty list for the current minimal OpenVINO runner.
        """

        return []
