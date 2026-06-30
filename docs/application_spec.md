# アプリケーション仕様書

## 目的

`trt_profiler` は、1つの ONNX モデルを複数の推論 backend / precision で実行し、変換や最適化によって精度が変化していないかを評価する Python ツールです。

主な比較対象は次です。

- ONNX Runtime CPU
- ONNX Runtime CUDA
- ONNX Runtime TensorRT Execution Provider
- OpenVINO CPU
- native TensorRT FP32 engine
- native TensorRT FP16 engine

1回の評価実行では、1つの論理モデルだけを扱います。複数 backend は同じモデルの variant として扱い、同一入力に対して raw 出力と postprocess 後の結果を比較します。

## 対象ユースケース

- TensorRT FP32 / FP16 化後に精度が大きく変わっていないか確認する
- ONNX Runtime と OpenVINO の出力差分を確認する
- ONNX Runtime TensorRT EP と native TensorRT の差分を確認する
- 分類モデルの top-k 一致率や score 差分を見る
- 検出モデルの box IoU、簡易 mAP、confidence 差分を見る
- CNN backbone の複数 feature map を layer ごとに比較する

## CLI 仕様

### 評価実行

```bash
trt-profiler eval -c path/to/config.yaml
```

比較ペアを CLI で上書きできます。

```bash
trt-profiler eval -c path/to/config.yaml --all-combinations
trt-profiler eval -c path/to/config.yaml --reference-to-all
```

### 既存 report から静的 HTML dashboard を生成

```bash
trt-profiler dashboard path/to/report.json -o path/to/dashboard.html
```

### Dash server で複数 report を横断表示

```bash
trt-profiler dash report1.json report2.json --host 127.0.0.1 --port 8050
```

Dash 版では report / comparison / stage / metric の絞り込み、DataTable、グラフ、comparison matrix、metric help modal、source image preview、heatmap preview、overlay preview を確認できます。

## 入力データ

標準 loader は次に対応します。

| type | 用途 |
| --- | --- |
| `npz_folder` / `npz` | `.npz` ファイル群を入力にする |
| `image_folder` / `image` | 画像フォルダーを入力にする |
| `video_file` / `video` | `cv2.VideoCapture` で動画を frame 単位に読む |

manifest 形式など、上記 preset にない dataset は詳細 config で `class` を指定して追加できます。

各 loader は `Sample` を返します。`Sample` は `id`, `data`, `source_path`, `label`, `annotations`, `metadata` を持ち、preprocess / metric / dashboard preview で利用されます。

## 評価パイプライン

```text
config load
  -> artifact build
  -> runner load
  -> dataset iteration
  -> preprocess
  -> input mapping
  -> inference
  -> output mapping
  -> raw metric
  -> postprocess
  -> post metric
  -> report
```

評価コアはタスク固有の意味を持ちません。分類、検出、feature map 比較などの意味は postprocessor と metric が担います。

## Config 仕様

config は2種類あります。

### 詳細 config

`common` セクションを持つ config は、pipeline が直接読む詳細 config として扱います。

主な構成:

- `common.model`: source ONNX モデル
- `common.variants`: backend variant と builder / runner
- `common.dataset`: dataset loader
- `common.comparisons`: 比較ペア
- `common.comparison_mode`: 比較ペア生成モード
- `common.input_mapping`: 共通入力名から backend 入力名への mapping
- `common.output_mapping`: backend 出力名から共通出力名への mapping
- `common.metrics`: raw / post metric
- `common.report`: reporter
- `preprocess`: preprocessor class
- `postprocessors`: postprocessor class list

### 簡易 config

`common` がない config は簡易 config として読み込まれ、内部で詳細 config に変換されます。

load の流れ:

```text
YAML
  -> parser.py: EvaluationConfig dataclass に変換
  -> validation.py: 必須項目を検証
  -> presets.py: preset を class path / runner config に解決
  -> builder.py: pipeline 用詳細 config に変換
```

最小例:

```yaml
model:
  name: my_model
  path: models/model.onnx

input:
  type: npz
  path: data/npz
  input_name: input_tensor
  backend_name: input

outputs:
  logits: output

variants:
  - ort_cpu
  - openvino_cpu

compare: reference-to-all

metrics:
  raw:
    - tensor_diff:
        outputs: [logits]

report:
  output_dir: reports/my_model
  formats: [json, dashboard, csv]
```

## Variant preset

標準 preset:

| preset | 内容 |
| --- | --- |
| `ort_cpu` | ONNX Runtime CPUExecutionProvider |
| `ort_cuda` | ONNX Runtime CUDAExecutionProvider + CPU fallback |
| `ort_trt` | ONNX Runtime TensorRT EP + CUDA + CPU fallback |
| `openvino_cpu` | OpenVINO CPU |
| `trt_fp32` | native TensorRT FP32 engine |
| `trt_fp16` | native TensorRT FP16 engine |

TensorRT preset は `tensor_rt` / `tensorrt` セクションで build option を共有できます。

```yaml
tensor_rt:
  artifacts_dir: artifacts
  trtexec: /usr/src/tensorrt/bin/trtexec
```

## Preprocess / Postprocess

標準 preprocessor:

- `NpzPreprocessor`
- `ImageNetPreprocessor`

標準 postprocessor:

- `IdentityPostprocessor`
- `SoftmaxPostprocessor`

独自 class を使う場合は dotted path を指定します。

```yaml
preprocess:
  class: my_project.preprocess.MyPreprocessor
  config:
    size: [640, 640]
    input_name: images
```

```yaml
postprocess:
  class: my_project.postprocess.MyPostprocessor
  config:
    score_threshold: 0.25
```

pipeline は `config` の中身を解釈しません。各具象 class が自身の config を解釈します。

## 評価 metric

### Raw / 汎用

- `TensorDiffMetric`
  - shape match
  - max / mean absolute error
  - RMSE
  - relative error
  - cosine similarity
  - allclose rate
  - NaN / Inf count
  - percentile error

### Feature map

- `FeatureMapDiffMetric`
  - layer ごとの tensor diff
  - layer ごとの cosine similarity
  - channelwise cosine
  - worst layer / worst sample
  - spatial heatmap `.npy` 保存

### Classification

- `ClassificationConsistencyMetric`
  - top-k 一致率
  - KL divergence
  - JS divergence
  - softmax score 差分
  - top-k ranking 一致率

- `ClassificationAccuracyMetric`
  - label あり accuracy
  - top-k accuracy

### Detection

- `DetectionConsistencyMetric`
  - reference / target detection 対応付け
  - box IoU
  - class 一致率
  - confidence 差分
  - NMS 後 box 数差分

- `DetectionAccuracyMetric`
  - annotation に対する簡易 mAP
  - class 別 / 閾値別の検出評価

## 出力仕様

標準 reporter:

| reporter | 出力 |
| --- | --- |
| `JsonReporter` | `report.json` |
| `CsvReporter` | `csv/*.csv` |
| `PlotlyDashboardReporter` | `dashboard.html` |

`report.json` は Dash server の入力にもなります。

CSV は次のような表を出力します。

- `metric_summary.csv`
- `per_sample.csv`
- `per_output.csv`
- `per_layer.csv`
- `worst_cases.csv`
- `failed_cases.csv`

## 拡張ポイント

| 目的 | 実装する基底 class |
| --- | --- |
| 新しい入力形式 | `DatasetLoader` |
| 新しい前処理 | `Preprocessor` |
| 新しい後処理 | `Postprocessor` |
| 新しい backend | `ArtifactBuilder` と `ModelRunner` |
| 新しい評価指標 | `Metric` |
| 新しい出力形式 | `Reporter` |

各 class は `trt_profiler.core.types` の契約に従います。実装後は config の `class` に dotted path を指定すれば利用できます。
