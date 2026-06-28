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

Generate a dashboard from an existing JSON report:

```bash
trt-profiler dashboard examples/squeezenet/reports_ort_openvino/report.json \
  -o examples/squeezenet/reports_ort_openvino/dashboard.html
```

## Full backend variant config

`config_full_variants.yaml` defines the intended full matrix:

- ORT CPU
- ORT CUDA
- ORT TensorRT EP
- OpenVINO CPU
- native TensorRT FP32
- native TensorRT FP16

Native TensorRT support targets TensorRT v11 only. Engine build uses `trtexec`;
native inference uses TensorRT v11 tensor APIs and `cuda-python` for CUDA buffer
management.

Run the native TensorRT FP32 vs FP16 example:

```bash
python3 -m pip install -e ".[tensorrt,dashboard]"
# Ensure the TensorRT SDK binary directory containing trtexec is on PATH.
trt-profiler eval -c examples/squeezenet/config_trt_fp32_fp16.yaml
```
