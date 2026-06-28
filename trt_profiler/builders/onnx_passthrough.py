"""ONNX passthrough artifact builder."""

from __future__ import annotations

from trt_profiler.core.types import ArtifactBuilder, ModelArtifact, SourceModel


class OnnxPassthroughBuilder(ArtifactBuilder):
    """Use the source ONNX file directly as the runtime artifact."""

    def build(self, source_model: SourceModel) -> ModelArtifact:
        """Return the source ONNX model as the artifact.

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
