from __future__ import annotations

from collections.abc import Iterator

from trt_profiler.core.types import DatasetLoader, Sample


class ManifestLoader(DatasetLoader):
    def __iter__(self) -> Iterator[Sample]:
        raise NotImplementedError("ManifestLoader will be implemented in the next phase.")
