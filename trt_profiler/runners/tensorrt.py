"""TensorRT v11 inference runner.

CUDA buffer allocation, tensor address binding, stream execution, and device to
host transfers are intentionally kept inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trt_profiler.core.types import ModelRunner, TensorDict, TensorSpec


@dataclass(frozen=True)
class _TensorInfo:
    name: str
    dtype: np.dtype[Any]
    is_input: bool


@dataclass
class _DeviceAllocation:
    ptr: int
    nbytes: int


class TensorRTRunner(ModelRunner):
    """Native TensorRT v11 runner backed by cuda-python.

    CUDA memory management is intentionally contained inside this runner. The
    rest of the evaluation pipeline only sees TensorDict inputs and outputs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._trt: Any | None = None
        self._cudart: Any | None = None
        self._engine: Any | None = None
        self._context: Any | None = None
        self._stream: int | None = None
        self._allocations: list[_DeviceAllocation] = []

    def load(self) -> None:
        """Deserialize the TensorRT engine and create CUDA resources.

        Raises
        ------
        FileNotFoundError
            If the configured engine path does not exist.
        RuntimeError
            If TensorRT, cuda-python, or TensorRT v11 runtime resources are not
            available.
        ValueError
            If the artifact path is missing.
        """

        if self.artifact.path is None:
            raise ValueError(f"TensorRT engine path is missing for {self.name}")
        if not self.artifact.path.exists():
            raise FileNotFoundError(self.artifact.path)

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt is not installed. Install TensorRT Python bindings to use "
                "native TensorRT inference."
            ) from exc

        try:
            from cuda.bindings import runtime as cudart
        except ImportError as exc:
            raise RuntimeError(
                "cuda-python is not installed. Install trt-profiler[tensorrt] to use "
                "native TensorRT inference."
            ) from exc

        self._trt = trt
        self._cudart = cudart
        _ensure_tensorrt_v11(str(trt.__version__))

        logger_level = getattr(trt.Logger, str(self.config.get("logger_level", "WARNING")))
        logger = trt.Logger(logger_level)
        runtime = trt.Runtime(logger)
        engine_bytes = self.artifact.path.read_bytes()
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.artifact.path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self._stream = _cuda_call(cudart.cudaStreamCreate)

    def infer(self, inputs: TensorDict) -> TensorDict:
        """Run TensorRT inference.

        Parameters
        ----------
        inputs
            Backend input tensor dictionary.

        Returns
        -------
        TensorDict
            Output tensors copied back to host memory.

        Raises
        ------
        RuntimeError
            If the runner has not been loaded.
        """

        if self._engine is None or self._context is None or self._stream is None:
            raise RuntimeError(f"Runner is not loaded: {self.name}")
        if self._cudart is None:
            raise RuntimeError("CUDA runtime is not initialized.")

        self._free_allocations()
        tensors = self._tensor_infos()
        input_infos = {tensor.name: tensor for tensor in tensors if tensor.is_input}
        output_infos = [tensor for tensor in tensors if not tensor.is_input]

        normalized_inputs = {
            name: self._normalize_input_array(name, inputs[name], input_infos[name].dtype)
            for name in input_infos
        }
        self._set_input_shapes(normalized_inputs)
        return self._infer_tensor_api(normalized_inputs, output_infos)

    def get_input_specs(self) -> list[TensorSpec]:
        """Return TensorRT input metadata.

        Returns
        -------
        list[TensorSpec]
            Input tensor metadata, or an empty list before loading.
        """

        if self._engine is None:
            return []
        return [
            TensorSpec(name=info.name, shape=self._shape_for(info.name), dtype=str(info.dtype))
            for info in self._tensor_infos()
            if info.is_input
        ]

    def get_output_specs(self) -> list[TensorSpec]:
        """Return TensorRT output metadata.

        Returns
        -------
        list[TensorSpec]
            Output tensor metadata, or an empty list before loading.
        """

        if self._engine is None:
            return []
        return [
            TensorSpec(name=info.name, shape=self._shape_for(info.name), dtype=str(info.dtype))
            for info in self._tensor_infos()
            if not info.is_input
        ]

    def close(self) -> None:
        """Release CUDA allocations and stream resources."""

        self._free_allocations()
        if self._stream is not None and self._cudart is not None:
            _cuda_call(self._cudart.cudaStreamDestroy, self._stream)
            self._stream = None

    def _infer_tensor_api(
        self,
        inputs: dict[str, np.ndarray],
        output_infos: list[_TensorInfo],
    ) -> TensorDict:
        assert self._context is not None
        assert self._cudart is not None
        assert self._stream is not None

        outputs: TensorDict = {}
        for name, array in inputs.items():
            allocation = self._malloc(array.nbytes)
            _cuda_call(
                self._cudart.cudaMemcpyAsync,
                allocation.ptr,
                array.ctypes.data,
                array.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )
            self._context.set_tensor_address(name, allocation.ptr)

        for info in output_infos:
            shape = tuple(int(dim) for dim in self._context.get_tensor_shape(info.name))
            output = np.empty(shape, dtype=info.dtype)
            allocation = self._malloc(output.nbytes)
            self._context.set_tensor_address(info.name, allocation.ptr)
            outputs[info.name] = output

        ok = self._context.execute_async_v3(stream_handle=self._stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")

        for name, output in outputs.items():
            address = int(self._context.get_tensor_address(name))
            _cuda_call(
                self._cudart.cudaMemcpyAsync,
                output.ctypes.data,
                address,
                output.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )

        _cuda_call(self._cudart.cudaStreamSynchronize, self._stream)
        return outputs

    def _tensor_infos(self) -> list[_TensorInfo]:
        assert self._engine is not None
        assert self._trt is not None
        if not hasattr(self._engine, "num_io_tensors"):
            raise RuntimeError("TensorRTRunner supports TensorRT v11 tensor API only.")
        infos: list[_TensorInfo] = []
        for index in range(int(self._engine.num_io_tensors)):
            name = str(self._engine.get_tensor_name(index))
            mode = self._engine.get_tensor_mode(name)
            infos.append(
                _TensorInfo(
                    name=name,
                    dtype=np.dtype(self._trt.nptype(self._engine.get_tensor_dtype(name))),
                    is_input=mode == self._trt.TensorIOMode.INPUT,
                )
            )
        return infos

    def _normalize_input_array(
        self,
        name: str,
        value: np.ndarray,
        dtype: np.dtype[Any],
    ) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"TensorRT input {name!r} must be a numpy array.")
        return np.ascontiguousarray(value, dtype=dtype)

    def _set_input_shapes(self, inputs: dict[str, np.ndarray]) -> None:
        assert self._engine is not None
        assert self._context is not None
        for name, array in inputs.items():
            self._context.set_input_shape(name, tuple(int(dim) for dim in array.shape))

    def _shape_for(self, name: str) -> tuple[int | str | None, ...]:
        if self._engine is None:
            return ()
        return tuple(self._engine.get_tensor_shape(name))

    def _malloc(self, nbytes: int) -> _DeviceAllocation:
        if self._cudart is None:
            raise RuntimeError("CUDA runtime is not initialized.")
        ptr = int(_cuda_call(self._cudart.cudaMalloc, nbytes))
        allocation = _DeviceAllocation(ptr=ptr, nbytes=nbytes)
        self._allocations.append(allocation)
        return allocation

    def _free_allocations(self) -> None:
        if self._cudart is None:
            self._allocations.clear()
            return
        while self._allocations:
            allocation = self._allocations.pop()
            _cuda_call(self._cudart.cudaFree, allocation.ptr)


def _cuda_call(func: Any, *args: Any) -> Any:
    result = func(*args)
    if not isinstance(result, tuple):
        return result
    error = result[0]
    raw_error_code = getattr(error, "value", error)
    if raw_error_code is None:
        raise RuntimeError(f"CUDA call failed: {func.__name__} returned {error}")
    error_code = int(raw_error_code)
    if error_code != 0:
        raise RuntimeError(f"CUDA call failed: {func.__name__} returned {error}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def _ensure_tensorrt_v11(version: str) -> None:
    major = int(version.split(".", maxsplit=1)[0])
    if major != 11:
        raise RuntimeError(
            f"TensorRTRunner supports TensorRT v11 only, but installed version is {version}."
        )
