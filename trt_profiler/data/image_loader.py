"""Image folder dataset loader."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from trt_profiler.core.types import DatasetLoader, Sample


class ImageFolderLoader(DatasetLoader):
    """Load image file paths from a directory.

    Notes
    -----
    The loader yields paths as sample data. Image decoding is handled by the
    configured preprocessor.

    Config Keys
    -----------
    path : str
        Root image directory.
    extensions : list[str], optional
        File extensions to include. Defaults to ``[".jpg", ".png"]``.
    limit : int, optional
        Maximum number of images to yield.
    """

    def __iter__(self) -> Iterator[Sample]:
        """Yield image samples.

        Yields
        ------
        Sample
            Sample containing the image path.
        """

        root = Path(str(self.config["path"]))
        extensions = [str(item).lower() for item in self.config.get("extensions", [".jpg", ".png"])]
        limit = self.config.get("limit")

        paths = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in extensions
        ]
        if limit is not None:
            paths = paths[: int(limit)]

        for path in paths:
            yield Sample(
                id=path.stem,
                data=path,
                source_path=path,
                metadata={"dataset_type": "image_folder"},
            )
