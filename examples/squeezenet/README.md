# SqueezeNet ONNX Example

This example downloads a small public ONNX model and creates a synthetic NPZ
input for a real ONNX Runtime smoke evaluation.

The model is SqueezeNet 1.1 from the ONNX Model Zoo.

## Prepare assets

```bash
python3 examples/squeezenet/prepare_assets.py
```

This creates:

```text
examples/squeezenet/assets/model.onnx
examples/squeezenet/inputs/sample_0001.npz
examples/squeezenet/images/sample_0001.png
examples/squeezenet/videos/sample.mp4
```

## Run

Install the runtime dependency first:

```bash
python3 -m pip install -e ".[onnxruntime]"
```

Then run:

```bash
trt-profiler eval -c examples/squeezenet/config.yaml
```

The report is written to:

```text
examples/squeezenet/reports/report.json
```

## ORT vs OpenVINO

Install OpenVINO and run:

```bash
python3 -m pip install -e ".[onnxruntime,openvino]"
trt-profiler eval -c examples/squeezenet/config_ort_openvino.yaml
```

Image-folder input is also available:

```bash
trt-profiler eval -c examples/squeezenet/config_image_ort_openvino.yaml
```

Video input via `cv2.VideoCapture` is available with the `video` extra:

```bash
python3 -m pip install -e ".[onnxruntime,openvino,video]"
trt-profiler eval -c examples/squeezenet/config_video_ort_openvino.yaml
```

## Full backend variant config

`config_full_variants.yaml` defines the intended full matrix:

- ORT CPU
- ORT CUDA
- ORT TensorRT EP
- OpenVINO CPU
- native TensorRT FP32
- native TensorRT FP16

Native TensorRT engine build uses `trtexec` when available. Native TensorRT
inference still requires the TensorRT runner implementation and CUDA bindings.
