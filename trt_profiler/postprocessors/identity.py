"""Identity postprocessor implementation."""

from __future__ import annotations

from trt_profiler.core.types import ComparableDict, Postprocessor, Sample, TensorDict


class IdentityPostprocessor(Postprocessor):
    """Return raw outputs unchanged as comparable values."""

    def __call__(self, outputs: TensorDict, sample: Sample | None = None) -> ComparableDict:
        """Return a shallow copy of raw outputs.

        Parameters
        ----------
        outputs
            Raw output tensor dictionary.
        sample
            Optional sample. It is accepted for interface compatibility and is
            not used.

        Returns
        -------
        ComparableDict
            Output dictionary copied from ``outputs``.
        """

        return {key: value for key, value in outputs.items()}
