"""OpenVINO artifact builder."""

from __future__ import annotations

from trt_profiler.core.types import ArtifactBuilder, ModelArtifact, SourceModel


class OpenVinoBuilder(ArtifactBuilder):
    """Prepare an OpenVINO artifact.

    MVP behavior keeps the ONNX path and lets OpenVinoRunner compile it. A later
    implementation can materialize IR files and populate artifact.path with them.
    """

    def build(self, source_model: SourceModel) -> ModelArtifact:
        """Prepare an artifact for OpenVINO.

        Parameters
        ----------
        source_model
            Source ONNX model definition.

        Returns
        -------
        ModelArtifact
            Artifact pointing at the source model path.
        """

        return ModelArtifact(
            variant_name=self.name,
            backend=self.backend,
            precision=self.precision,
            path=source_model.path,
            config={"source_format": source_model.format, **self.config},
        )
