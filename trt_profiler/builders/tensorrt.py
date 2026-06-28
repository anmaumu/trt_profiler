"""TensorRT v11 engine builder.

This module builds TensorRT engines with ``trtexec`` and prepares FP16 ONNX
models for TensorRT v11 strongly typed networks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from importlib import import_module
from pathlib import Path

from trt_profiler.core.types import ArtifactBuilder, ModelArtifact, SourceModel


class TensorRTBuilder(ArtifactBuilder):
    """Build or reuse a TensorRT engine.

    TensorRT v11 support builds engines through trtexec.
    """

    def build(self, source_model: SourceModel) -> ModelArtifact:
        """Build or reuse a TensorRT engine artifact.

        Parameters
        ----------
        source_model
            Source ONNX model definition.

        Returns
        -------
        ModelArtifact
            TensorRT engine artifact.

        Raises
        ------
        ValueError
            If required build configuration is missing or unsupported.
        RuntimeError
            If ``trtexec`` cannot be found or the build command fails.
        """

        engine_path = self.config.get("engine_path")
        artifact_path = Path(str(engine_path)) if engine_path is not None else None
        should_build = bool(self.config.get("build", False))
        if should_build:
            if artifact_path is None:
                raise ValueError("TensorRTBuilder requires config.engine_path when build=true.")
            builder_backend = str(self.config.get("builder_backend", "trtexec")).lower()
            if builder_backend != "trtexec":
                raise ValueError("TensorRT v11 builds are supported through trtexec only.")
            onnx_path = self._prepare_onnx_for_precision(source_model.path, artifact_path)
            self._build_with_trtexec(onnx_path, artifact_path)
        return ModelArtifact(
            variant_name=self.name,
            backend=self.backend,
            precision=self.precision,
            path=artifact_path,
            config={"source_model": str(source_model.path), **self.config},
        )

    def _build_with_trtexec(self, onnx_path: Path, engine_path: Path) -> None:
        trtexec_config = str(self.config.get("trtexec", "trtexec"))
        trtexec = (
            str(Path(trtexec_config))
            if Path(trtexec_config).exists()
            else shutil.which(trtexec_config)
        )
        if trtexec is None:
            raise RuntimeError("trtexec was not found on PATH.")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]
        if bool(self.config.get("skip_inference", True)):
            command.append("--skipInference")

        workspace_size = self.config.get("workspace_size")
        if workspace_size is not None:
            command.append(f"--memPoolSize=workspace:{workspace_size}")

        extra_args = self.config.get("extra_args", [])
        command.extend(str(arg) for arg in extra_args)

        subprocess.run(command, check=True, env=self._trtexec_env())

    def _prepare_onnx_for_precision(self, onnx_path: Path, engine_path: Path) -> Path:
        precision = str(self.config.get("precision", self.precision or "fp32")).lower()
        if precision == "fp32":
            return onnx_path
        if precision != "fp16":
            raise ValueError(
                "TensorRT v11 trtexec builder supports fp32 and fp16. "
                "For int8, provide a pre-quantized ONNX model."
            )

        fp16_onnx_path = Path(
            str(self.config.get("fp16_onnx_path", engine_path.with_suffix(".fp16.onnx")))
        )
        if fp16_onnx_path.exists() and not bool(self.config.get("reconvert_fp16_onnx", False)):
            return fp16_onnx_path

        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError as exc:
            raise RuntimeError(
                "FP16 TensorRT v11 builds require onnx and onnxconverter-common. "
                "Install trt-profiler[tensorrt]."
            ) from exc

        fp16_onnx_path.parent.mkdir(parents=True, exist_ok=True)
        model = onnx.load(str(onnx_path))
        keep_io_types = bool(self.config.get("keep_io_types", True))
        converted = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
        onnx.save(converted, str(fp16_onnx_path))
        return fp16_onnx_path

    def _trtexec_env(self) -> dict[str, str]:
        env = dict(os.environ)
        extra_library_paths = [str(path) for path in self.config.get("library_paths", [])]
        tensorrt_libs = _find_python_tensorrt_libs()
        if tensorrt_libs is not None:
            extra_library_paths.append(str(tensorrt_libs))
        if extra_library_paths:
            current = env.get("LD_LIBRARY_PATH")
            if current:
                extra_library_paths.append(current)
            env["LD_LIBRARY_PATH"] = ":".join(extra_library_paths)
        return env


def _find_python_tensorrt_libs() -> Path | None:
    try:
        tensorrt_libs = import_module("tensorrt_libs")
    except ImportError:
        return None
    paths = list(getattr(tensorrt_libs, "__path__", []))
    if not paths:
        return None
    return Path(str(paths[0]))
