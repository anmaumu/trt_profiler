# SqueezeNet ONNX サンプル

このサンプルは、ONNX Model Zoo の SqueezeNet 1.1 を使って `trt_profiler` の実行方法を確認するためのものです。

すべての `config*.yaml` は簡易 config 形式で書かれています。`load_config()` が内部で詳細 config に展開してから評価を実行します。

## Asset 作成

```bash
python3 examples/squeezenet/prepare_assets.py
```

作成される主なファイル:

```text
examples/squeezenet/assets/model.onnx
examples/squeezenet/inputs/sample_0001.npz
examples/squeezenet/images/sample_0001.png
examples/squeezenet/videos/sample.mp4
```

## Config 一覧

| config | 内容 |
| --- | --- |
| `config.yaml` | ORT CPU vs ORT CPU の最小 smoke test |
| `config_simple.yaml` | ORT CPU vs OpenVINO CPU、softmax postprocess、CSV/dashboard 出力 |
| `config_postprocess.yaml` | ORT CPU vs ORT CPU、classification post metric |
| `config_ort_openvino.yaml` | NPZ 入力で ORT CPU vs OpenVINO CPU |
| `config_image_ort_openvino.yaml` | 画像入力で ORT CPU vs OpenVINO CPU |
| `config_video_ort_openvino.yaml` | 動画入力で ORT CPU vs OpenVINO CPU |
| `config_ort_trt_fp32.yaml` | ORT CPU vs native TensorRT FP32 |
| `config_trt_fp32_fp16.yaml` | native TensorRT FP32 vs FP16 |
| `config_full_variants.yaml` | ORT / OpenVINO / TensorRT の比較 matrix 例 |

## 最小実行

```bash
python3 -m pip install -e ".[onnxruntime]"
trt-profiler eval -c examples/squeezenet/config.yaml
```

出力:

```text
examples/squeezenet/reports/report.json
```

## ORT vs OpenVINO

```bash
python3 -m pip install -e ".[onnxruntime,openvino,dashboard]"
trt-profiler eval -c examples/squeezenet/config_ort_openvino.yaml
```

画像入力:

```bash
trt-profiler eval -c examples/squeezenet/config_image_ort_openvino.yaml
```

動画入力:

```bash
python3 -m pip install -e ".[onnxruntime,openvino,video,dashboard]"
trt-profiler eval -c examples/squeezenet/config_video_ort_openvino.yaml
```

## Postprocess metric

`config_postprocess.yaml` は raw logits を `SoftmaxPostprocessor` で確率に変換し、post stage で `ClassificationConsistencyMetric` を評価します。

```bash
trt-profiler eval -c examples/squeezenet/config_postprocess.yaml
```

## TensorRT

native TensorRT runner は TensorRT v11 を対象にしています。engine build は `trtexec`、推論は TensorRT v11 tensor API と `cuda-python` を使います。

```bash
python3 -m pip install -e ".[tensorrt,dashboard]"
# trtexec が PATH 上にあることを確認してください。
trt-profiler eval -c examples/squeezenet/config_trt_fp32_fp16.yaml
```

`trtexec` が PATH 上にない場合は config に追加します。

```yaml
tensor_rt:
  artifacts_dir: examples/squeezenet/artifacts
  trtexec: /path/to/trtexec
```

## Dashboard

既存の `report.json` から静的 HTML を作る場合:

```bash
trt-profiler dashboard examples/squeezenet/reports_ort_openvino/report.json \
  -o examples/squeezenet/reports_ort_openvino/dashboard.html
```

Dash server で見る場合:

```bash
trt-profiler dash examples/squeezenet/reports_ort_openvino/report.json
```
