"""Classification postprocessors."""

from __future__ import annotations

import numpy as np

from trt_profiler.core.types import ComparableDict, Postprocessor, Sample, TensorDict


class SoftmaxPostprocessor(Postprocessor):
    """Convert logits to probabilities.

    Config Keys
    -----------
    logits_key : str, optional
        Raw output key containing logits. Defaults to ``"logits"``.
    probs_key : str, optional
        Output key for probabilities. Defaults to ``"probs"``.
    axis : int, optional
        Axis used for softmax. Defaults to ``-1``.
    squeeze : bool, optional
        Squeeze singleton dimensions before returning probabilities. Defaults
        to ``True``.
    """

    def __call__(self, outputs: TensorDict, sample: Sample | None = None) -> ComparableDict:
        """Convert logits to probabilities.

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
            Dictionary containing softmax probabilities.
        """

        del sample
        logits_key = str(self.config.get("logits_key", "logits"))
        probs_key = str(self.config.get("probs_key", "probs"))
        axis = int(self.config.get("axis", -1))
        logits = np.asarray(outputs[logits_key], dtype=np.float64)
        shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=axis, keepdims=True)
        if bool(self.config.get("squeeze", True)):
            probs = np.squeeze(probs)
        return {probs_key: probs.astype(np.float32)}
