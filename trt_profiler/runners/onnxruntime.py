"""ONNX Runtime inference runner."""

from __future__ import annotations

from typing import Any

from trt_profiler.core.types import ModelRunner, TensorDict, TensorSpec


class OnnxRuntimeRunner(ModelRunner):
    """Run inference with ONNX Runtime.

    Config Keys
    -----------
    providers : list[str], optional
        ONNX Runtime execution provider names.
    provider_options : dict, optional
        Provider-specific options. A mapping keyed by provider name is converted
        to ONNX Runtime's ordered provider options list.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._session: Any | None = None

    def load(self) -> None:
        """Create the ONNX Runtime inference session.

        Raises
        ------
        RuntimeError
            If ONNX Runtime is not installed.
        ValueError
            If the artifact path is missing.
        """

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install trt-profiler[onnxruntime]."
            ) from exc

        if self.artifact.path is None:
            raise ValueError(f"ONNX Runtime artifact path is missing for {self.name}")

        providers = self.config.get("providers")
        provider_options = self.config.get("provider_options")
        kwargs: dict[str, Any] = {}
        if providers is not None:
            kwargs["providers"] = providers
        if provider_options is not None:
            kwargs["provider_options"] = _normalize_provider_options(providers, provider_options)

        self._session = ort.InferenceSession(str(self.artifact.path), **kwargs)

    def infer(self, inputs: TensorDict) -> TensorDict:
        """Run ONNX Runtime inference.

        Parameters
        ----------
        inputs
            Backend input tensor dictionary.

        Returns
        -------
        TensorDict
            Output tensors keyed by ONNX Runtime output names.
        """

        if self._session is None:
            raise RuntimeError(f"Runner is not loaded: {self.name}")

        output_names = [output.name for output in self._session.get_outputs()]
        values = self._session.run(output_names, inputs)
        return dict(zip(output_names, values, strict=True))

    def get_input_specs(self) -> list[TensorSpec]:
        """Return ONNX Runtime input metadata.

        Returns
        -------
        list[TensorSpec]
            Input tensor metadata, or an empty list before loading.
        """

        if self._session is None:
            return []
        return [
            TensorSpec(name=item.name, shape=tuple(item.shape), dtype=str(item.type))
            for item in self._session.get_inputs()
        ]

    def get_output_specs(self) -> list[TensorSpec]:
        """Return ONNX Runtime output metadata.

        Returns
        -------
        list[TensorSpec]
            Output tensor metadata, or an empty list before loading.
        """

        if self._session is None:
            return []
        return [
            TensorSpec(name=item.name, shape=tuple(item.shape), dtype=str(item.type))
            for item in self._session.get_outputs()
        ]


def _normalize_provider_options(providers: object, provider_options: object) -> object:
    if not isinstance(provider_options, dict) or not isinstance(providers, list):
        return provider_options
    return [provider_options.get(provider, {}) for provider in providers]
