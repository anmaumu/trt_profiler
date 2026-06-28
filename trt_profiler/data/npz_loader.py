from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from trt_profiler.core.types import DatasetLoader, Sample


class NpzFolderLoader(DatasetLoader):
    def __iter__(self) -> Iterator[Sample]:
        root = Path(str(self.config["path"]))
        pattern = str(self.config.get("pattern", "*.npz"))
        limit = self.config.get("limit")
        paths = sorted(root.glob(pattern))
        if limit is not None:
            paths = paths[: int(limit)]

        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            yield Sample(
                id=path.stem,
                data=arrays,
                source_path=path,
                metadata={"dataset_type": "npz_folder"},
            )
