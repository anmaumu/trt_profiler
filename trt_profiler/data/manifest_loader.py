"""Manifest dataset loader placeholder."""

from __future__ import annotations

from collections.abc import Iterator

from trt_profiler.core.types import DatasetLoader, Sample


class ManifestLoader(DatasetLoader):
    """Placeholder loader for manifest-based datasets.

    Notes
    -----
    This class reserves the extension point for future manifest formats.
    """

    def __iter__(self) -> Iterator[Sample]:
        """Yield manifest samples.

        Raises
        ------
        NotImplementedError
            Always raised until manifest support is implemented.
        """

        raise NotImplementedError("ManifestLoader will be implemented in the next phase.")
