from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from trt_profiler.core.types import ArtifactBuilder, ModelArtifact, SourceModel


class TensorRTBuilder(ArtifactBuilder):
    """Build or reuse a TensorRT engine.

    Uses trtexec when it is available and config.build is true. If build is
    false, it simply returns the configured engine path.
    """

    def build(self, source_model: SourceModel) -> ModelArtifact:
        engine_path = self.config.get("engine_path")
        artifact_path = Path(str(engine_path)) if engine_path is not None else None
        should_build = bool(self.config.get("build", False))
        if should_build:
            if artifact_path is None:
                raise ValueError("TensorRTBuilder requires config.engine_path when build=true.")
            self._build_with_trtexec(source_model.path, artifact_path)
        return ModelArtifact(
            variant_name=self.name,
            backend=self.backend,
            precision=self.precision,
            path=artifact_path,
            config={"source_model": str(source_model.path), **self.config},
        )

    def _build_with_trtexec(self, onnx_path: Path, engine_path: Path) -> None:
        trtexec = shutil.which(str(self.config.get("trtexec", "trtexec")))
        if trtexec is None:
            raise RuntimeError("trtexec was not found on PATH.")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]

        precision = str(self.config.get("precision", self.precision or "fp32")).lower()
        if precision == "fp16":
            command.append("--fp16")
        elif precision == "int8":
            command.append("--int8")

        workspace_size = self.config.get("workspace_size")
        if workspace_size is not None:
            command.append(f"--memPoolSize=workspace:{workspace_size}")

        extra_args = self.config.get("extra_args", [])
        command.extend(str(arg) for arg in extra_args)

        subprocess.run(command, check=True)
