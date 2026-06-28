"""Video dataset loader using OpenCV."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from trt_profiler.core.types import DatasetLoader, Sample


class VideoLoader(DatasetLoader):
    """Decode video frames with OpenCV.

    Config Keys
    -----------
    path : str
        Video file path or directory containing videos.
    extensions : list[str], optional
        File extensions used when ``path`` is a directory.
    frame_stride : int, optional
        Yield every Nth frame. Defaults to ``1``.
    max_frames_per_video : int, optional
        Maximum number of yielded frames per video.
    """

    def __iter__(self) -> Iterator[Sample]:
        """Yield decoded video frame samples.

        Yields
        ------
        Sample
            Sample whose data is a BGR frame ndarray from OpenCV.

        Raises
        ------
        RuntimeError
            If OpenCV is unavailable or a video cannot be opened.
        """

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python-headless is required for VideoLoader.") from exc

        paths = self._video_paths()
        frame_stride = int(self.config.get("frame_stride", 1))
        max_frames = self.config.get("max_frames_per_video")
        max_frames_int = int(max_frames) if max_frames is not None else None

        for path in paths:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {path}")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            yielded = 0
            frame_index = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % frame_stride == 0:
                        timestamp_ms = (
                            frame_index / fps * 1000.0
                            if fps > 0.0
                            else float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                        )
                        yield Sample(
                            id=f"{path.stem}:{frame_index}",
                            data=frame,
                            source_path=path,
                            metadata={
                                "dataset_type": "video",
                                "frame_index": frame_index,
                                "timestamp_ms": timestamp_ms,
                                "fps": fps,
                                "color_format": "BGR",
                            },
                        )
                        yielded += 1
                        if max_frames_int is not None and yielded >= max_frames_int:
                            break
                    frame_index += 1
            finally:
                cap.release()

    def _video_paths(self) -> list[Path]:
        path = Path(str(self.config["path"]))
        if path.is_file():
            return [path]

        extensions = [
            str(item).lower()
            for item in self.config.get("extensions", [".mp4", ".avi", ".mov", ".mkv"])
        ]
        return [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in extensions
        ]
