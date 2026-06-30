# ユーザーモデル評価チュートリアル

このチュートリアルでは、手元の ONNX モデルを `trt_profiler` で評価する流れを説明します。

## 前提

必要なもの:

- ONNX モデルファイル
- 評価に使う入力データ
- モデルの入力 tensor 名
- モデルの出力 tensor 名
- 必要に応じて独自 preprocess / postprocess class

環境構築は [README.md](../README.md) を参照してください。

CPU だけでまず動かす場合:

```bash
uv venv --python 3.12
uv pip install -e ".[onnxruntime,openvino,dashboard,dev]"
```

動画入力を使う場合:

```bash
uv pip install -e ".[video]"
```

native TensorRT を使う場合:

```bash
uv pip install -e ".[tensorrt]"
```

## 1. モデルの入出力名を確認する

ONNX モデルの入力名と出力名を確認します。Netron などを使ってもよいです。

例:

```text
input tensor: images
output tensor: logits
```

`trt_profiler` の config では、内部で扱う論理名と backend の実名を mapping します。

```yaml
input:
  input_name: input_tensor
  backend_name: images

outputs:
  logits: logits
```

左側の `logits` はツール内の論理名、右側の `logits` は backend が返す実出力名です。

## 2. 入力データを準備する

### NPZ 入力

最初の確認には NPZ が一番単純です。

```text
data/npz/
  sample_0001.npz
  sample_0002.npz
```

各 `.npz` に `data` という key で入力 tensor を保存した場合:

```yaml
input:
  type: npz
  path: data/npz
  input_name: input_tensor
  backend_name: images
  npz_key: data
  dtype: float32
```

### 画像入力

画像を直接読む場合:

```text
data/images/
  sample_0001.jpg
  sample_0002.jpg
```

```yaml
input:
  type: image
  path: data/images
  input_name: input_tensor
  backend_name: images

preprocess:
  type: imagenet
  input_name: input_tensor
  size: [224, 224]
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  layout: NCHW
  dtype: float32
```

### 動画入力

動画を frame 単位で読む場合:

```yaml
input:
  type: video
  path: data/video/sample.mp4
  input_name: input_tensor
  backend_name: images

preprocess:
  type: imagenet
  input_name: input_tensor
  size: [224, 224]
  layout: NCHW
  dtype: float32
```

## 3. まず ORT CPU vs OpenVINO CPU で動かす

最初は TensorRT を入れずに CPU backend で通すのがおすすめです。

`configs/my_model_cpu.yaml`:

```yaml
model:
  name: my_model
  path: models/my_model.onnx
  tasks: [classification]

input:
  type: npz
  path: data/npz
  input_name: input_tensor
  backend_name: images
  npz_key: data
  dtype: float32

outputs:
  logits: logits

variants:
  - ort_cpu
  - openvino_cpu

compare: reference-to-all

metrics:
  raw:
    - tensor_diff:
        outputs: [logits]
        atol: 1.0e-5
        rtol: 1.0e-5
        percentiles: [95, 99]

report:
  output_dir: reports/my_model_cpu
  formats: [json, dashboard, csv]
```

実行:

```bash
trt-profiler eval -c configs/my_model_cpu.yaml
```

出力:

```text
reports/my_model_cpu/
  report.json
  dashboard.html
  csv/
```

## 4. 分類モデルで postprocess 後も比較する

logits を softmax して、top-k 一致率や KL divergence / JS divergence を見ます。

```yaml
postprocess:
  type: softmax
  logits_key: logits
  probs_key: probs
  axis: -1
  squeeze: true

metrics:
  raw:
    - tensor_diff:
        outputs: [logits]
  post:
    - classification_consistency:
        probs_key: probs
        topk: [1, 5]
```

label 付き accuracy を見たい場合は、dataset loader が `Sample.label` を返す必要があります。詳細 config で manifest loader class を指定するか、独自 dataset loader を使って label を渡してください。

```yaml
metrics:
  post:
    - classification_accuracy:
        probs_key: probs
        topk: [1, 5]
```

## 5. 検出モデルを比較する

検出モデルでは、raw 出力を detection 形式に変換する postprocessor を用意するのが基本です。

postprocessor の戻り値は、metric config に合わせて次のような key を持たせます。

```python
{
    "boxes": np.ndarray,   # shape: [N, 4]
    "scores": np.ndarray,  # shape: [N]
    "labels": np.ndarray,  # shape: [N]
}
```

config 例:

```yaml
postprocess:
  class: my_project.postprocess.MyDetectionPostprocessor
  config:
    boxes_key: raw_boxes
    scores_key: raw_scores
    labels_key: raw_labels
    score_threshold: 0.25
    nms_iou: 0.45

metrics:
  post:
    - detection_consistency:
        boxes_key: boxes
        scores_key: scores
        labels_key: labels
        iou_threshold: 0.5
    - detection_accuracy:
        boxes_key: boxes
        scores_key: scores
        labels_key: labels
        iou_thresholds: [0.5, 0.75]
```

annotation に対する accuracy を見る場合は、dataset の `Sample.annotations` に ground truth を入れてください。

## 6. Feature map を複数 layer で比較する

backbone の中間 feature を ONNX 出力として expose している場合、複数出力を比較できます。

```yaml
outputs:
  layer1: backbone.layer1
  layer2: backbone.layer2
  layer3: backbone.layer3

metrics:
  raw:
    - feature_diff:
        outputs: [layer1, layer2, layer3]
        channel_axis: 1
        save_heatmaps: true
        heatmap_dir: reports/my_model_features/heatmaps
        percentiles: [95, 99]

report:
  output_dir: reports/my_model_features
  formats: [json, dashboard, csv]
```

`save_heatmaps: true` の場合、spatial mean absolute error の `.npy` heatmap が保存されます。Dash server では heatmap preview と source image overlay を確認できます。

## 7. TensorRT FP32 / FP16 を追加する

TensorRT 環境が使える場合、variant に `trt_fp32` と `trt_fp16` を追加します。

```yaml
variants:
  - ort_cpu
  - trt_fp32
  - trt_fp16

compare: reference-to-all

tensor_rt:
  artifacts_dir: artifacts/my_model
  trtexec: /usr/src/tensorrt/bin/trtexec
```

実行:

```bash
trt-profiler eval -c configs/my_model_trt.yaml
```

TensorRT v11 の FP16 build では、内部的に ONNX を FP16 化してから `trtexec` で engine を作ります。デフォルトでは比較しやすいように、モデル入出力は FP32 のまま維持します。

## 8. Dashboard を見る

静的 HTML:

```bash
open reports/my_model_cpu/dashboard.html
```

Dash server:

```bash
trt-profiler dash reports/my_model_cpu/report.json --host 127.0.0.1 --port 8050
```

複数 report を横断する場合:

```bash
trt-profiler dash \
  reports/my_model_cpu/report.json \
  reports/my_model_trt/report.json \
  --host 127.0.0.1 \
  --port 8050
```

## 9. 独自 preprocess を実装する

`Preprocessor` を継承します。

```python
from __future__ import annotations

import numpy as np

from trt_profiler.core.types import Sample, TensorDict
from trt_profiler.preprocessors.base import Preprocessor


class MyPreprocessor(Preprocessor):
    def __call__(self, sample: Sample) -> TensorDict:
        input_name = str(self.config.get("input_name", "input_tensor"))
        array = np.asarray(sample.data, dtype=np.float32)
        return {input_name: array}
```

config:

```yaml
preprocess:
  class: my_project.preprocess.MyPreprocessor
  config:
    input_name: input_tensor
```

## 10. 独自 postprocess を実装する

`Postprocessor` を継承します。

```python
from __future__ import annotations

import numpy as np

from trt_profiler.core.types import ComparableDict, Sample, TensorDict
from trt_profiler.postprocessors.base import Postprocessor


class MyPostprocessor(Postprocessor):
    def __call__(self, outputs: TensorDict, sample: Sample | None = None) -> ComparableDict:
        logits_key = str(self.config.get("logits_key", "logits"))
        probs_key = str(self.config.get("probs_key", "probs"))
        logits = outputs[logits_key]
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return {probs_key: exp / np.sum(exp, axis=-1, keepdims=True)}
```

config:

```yaml
postprocess:
  class: my_project.postprocess.MyPostprocessor
  config:
    logits_key: logits
    probs_key: probs
```

## 11. 独自 metric を実装する

`Metric` を継承し、sample ごとの `update()` と最後の `compute()` を実装します。

```python
from __future__ import annotations

from typing import Any

from trt_profiler.core.types import MetricSummaryRecord, Sample, SampleMetricRecord
from trt_profiler.metrics.base import Metric


class MyMetric(Metric):
    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self.values: list[float] = []

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        value = float(reference["score"] == target["score"])
        self.values.append(value)
        return [
            SampleMetricRecord(
                sample_id=sample.id if sample is not None else "",
                comparison="",
                stage="",
                metric=self.name,
                stat="match",
                value=value,
            )
        ]

    def compute(self) -> list[MetricSummaryRecord]:
        mean = sum(self.values) / len(self.values) if self.values else 0.0
        return [
            MetricSummaryRecord(
                comparison="",
                stage="",
                metric=self.name,
                stat="match_rate",
                value=mean,
            )
        ]
```

config:

```yaml
metrics:
  post:
    - name: my_metric
      class: my_project.metrics.MyMetric
      config: {}
```

## 12. よくある確認ポイント

- `input.backend_name` が ONNX の実入力名と一致しているか
- `outputs` の右側が backend の実出力名と一致しているか
- preprocess が返す key と `input.input_name` が一致しているか
- postprocess が参照する raw output key が `outputs` の左側の論理名と一致しているか
- `compare: reference-to-all` のとき、最初の variant が reference になっているか
- TensorRT 実行時に `trtexec` と TensorRT 共有ライブラリへ path が通っているか
