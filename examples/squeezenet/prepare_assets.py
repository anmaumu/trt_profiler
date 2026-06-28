from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
)


def main() -> None:
    root = Path(__file__).resolve().parent
    assets_dir = root / "assets"
    inputs_dir = root / "inputs"
    images_dir = root / "images"
    videos_dir = root / "videos"
    assets_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    model_path = assets_dir / "model.onnx"
    if not model_path.exists():
        with urlopen(MODEL_URL, timeout=120) as response:
            model_path.write_bytes(response.read())
        print(f"Downloaded {model_path} ({model_path.stat().st_size} bytes)")
    else:
        print(f"Using existing {model_path}")

    rng = np.random.default_rng(seed=0)
    sample = rng.normal(size=(1, 3, 224, 224)).astype(np.float32)
    input_path = inputs_dir / "sample_0001.npz"
    np.savez(input_path, data=sample)
    print(f"Wrote {input_path}")

    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; skipped sample image generation")
        return

    image = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    image_path = images_dir / "sample_0001.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    print(f"Wrote {image_path}")

    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed; skipped sample video generation")
        return

    video_path = videos_dir / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (224, 224),
    )
    for _ in range(12):
        frame_rgb = rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
        frame_bgr = frame_rgb[..., ::-1]
        writer.write(frame_bgr)
    writer.release()
    print(f"Wrote {video_path}")


if __name__ == "__main__":
    main()
