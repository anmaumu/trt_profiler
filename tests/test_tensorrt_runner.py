from __future__ import annotations

import pytest

from trt_profiler.runners.tensorrt import _cuda_call, _ensure_tensorrt_v11


def test_cuda_call_returns_single_payload_value() -> None:
    def fake_call() -> tuple[int, int]:
        return (0, 123)

    assert _cuda_call(fake_call) == 123


def test_cuda_call_raises_on_nonzero_error() -> None:
    def fake_call() -> tuple[int]:
        return (1,)

    with pytest.raises(RuntimeError, match="CUDA call failed"):
        _cuda_call(fake_call)


def test_ensure_tensorrt_v11_accepts_v11() -> None:
    _ensure_tensorrt_v11("11.1.0")


def test_ensure_tensorrt_v11_rejects_other_major_versions() -> None:
    with pytest.raises(RuntimeError, match="v11 only"):
        _ensure_tensorrt_v11("10.16.1")
