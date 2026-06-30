# trt_profiler

`trt_profiler` は、1つのONNXモデルを複数の推論バックエンドで実行し、
最適化や変換によって精度が変化していないかを評価するためのツールです。

主な比較対象は次のような組み合わせです。

- ONNX Runtime CPU vs ONNX Runtime CUDA
- ONNX Runtime CPU vs ONNX Runtime TensorRT Execution Provider
- ONNX Runtime CPU vs OpenVINO CPU
- TensorRT FP32 engine vs TensorRT FP16 engine

## 現在の対応範囲

入力データ:

- NPZファイル
- 画像フォルダー
- `cv2.VideoCapture` でデコードする動画ファイル

評価ステージ:

- 後処理前のraw tensor比較
- 後処理後の結果比較
- 分類モデル向けmetric
- 検出モデル向けmetric
- CNNバックボーンなどの複数feature map比較

レポート:

- JSON report
- PlotlyによるインタラクティブHTML dashboard

TensorRT native runnerはTensorRT v11のみを対象にしています。

## フォルダー構成

```text
trt_profiler/
  builders/        モデル変換やengine build
  runners/         backendごとの推論runner
  data/            NPZ、画像、動画、manifestのdataset loader
  preprocessors/   差し替え可能な前処理class
  postprocessors/  差し替え可能な後処理class
  metrics/         raw / postprocess metric
  report/          JSON / Plotly dashboard reporter
docs/
  README.md        ドキュメント目次
  application_spec.md
                   アプリケーション仕様
  uml.md           UML図
  tutorial_user_model.md
                   ユーザーモデル評価チュートリアル
  design.md        初期設計から詳細方針まで含む設計仕様
examples/
  squeezenet/      公開ONNXモデルを使う実行サンプル
tests/
```

## 実装ルール

このプロジェクトでは `ruff` と `mypy` のチェックに従います。

変更後は次を実行してください。

```bash
.venv/bin/ruff format trt_profiler tests examples/squeezenet
.venv/bin/ruff check trt_profiler tests examples/squeezenet
.venv/bin/mypy trt_profiler tests
.venv/bin/pytest -q
```

Windows PowerShellでは `.venv/bin/...` を `.venv\Scripts\...` に読み替えてください。

## WSL2 / Linux 環境構築

TensorRT native runnerまで含めた全backend比較を行う場合は、WSL2 Ubuntuまたは
NVIDIA GPU付きのLinux環境を推奨します。

### 1. 前提

推奨環境:

- WSL2 Ubuntu または native Linux
- NVIDIA GPU
- WSL2/Linuxから認識できるNVIDIA driver
- Python 3.10以上
- `uv`

GPUが見えているか確認します。

```bash
nvidia-smi
```

必要に応じて基本ツールを入れます。

```bash
sudo apt update
sudo apt install -y git curl build-essential
```

`uv` がない場合はインストールします。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール直後に `uv` が見つからない場合は、shellを開き直すかshell profileを読み直してください。

### 2. Python環境作成

```bash
uv venv --python 3.12
uv pip install -e ".[onnxruntime,openvino,video,dashboard,dev]"
```

TensorRTも使う場合は追加で入れます。

```bash
uv pip install -e ".[tensorrt]"
```

### 3. TensorRT v11 と trtexec

native TensorRT評価では次を使います。

- TensorRT Python package `>=11,<12`
- `cuda-python`
- TensorRT v11 tensor API
- engine build用の `trtexec`

TensorRT本体は、利用するUbuntu/CUDA/driverに合うNVIDIA公式手順で導入してください。
`trtexec` は多くの場合 `libnvinfer-bin` に含まれます。

`trtexec` がPATH上にある場合は次で確認できます。

```bash
trtexec --help | head
```

PATH上にない場合はconfigで明示します。

```yaml
builder:
  config:
    trtexec: /path/to/trtexec
```

NVIDIA apt repositoryが設定済みで、`sudo` が使えない環境では、debをローカル展開して使うこともできます。

```bash
mkdir -p .local_debs .local_tensorrt
cd .local_debs
apt-get download libnvinfer-bin
cd ..
dpkg-deb -x .local_debs/libnvinfer-bin_*.deb .local_tensorrt
```

この場合はconfigに次のように指定します。

```yaml
builder:
  config:
    trtexec: .local_tensorrt/usr/bin/trtexec
```

`TensorRTBuilder` は `trtexec` 実行時に、TensorRT Python wheel内の共有ライブラリパスを
`LD_LIBRARY_PATH` に自動追加します。

### 4. TensorRT v11 FP16の注意

TensorRT v11はstrongly typed networkを前提にしています。このため、このプロジェクトでは
古い `trtexec --fp16` 方式は使いません。

FP16 buildでは、まず `onnxconverter-common` を使ってONNXをFP16化し、そのFP16 ONNXを
`trtexec` に渡してengineを作ります。デフォルトでは比較しやすいように、モデルの入力/出力型は
FP32のまま維持します。

## Windows 環境構築

Windowsでは、ONNX Runtime CPU、OpenVINO CPU、画像/動画入力、dashboard生成を対象にできます。

native TensorRT評価は、現状ではWSL2/Linuxを推奨します。理由は、TensorRT pathをLinux版
`trtexec`、Linux共有ライブラリ、`LD_LIBRARY_PATH` の前提で検証しているためです。

### 1. 前提

インストールするもの:

- Git for Windows
- Python 3.10以上、または `uv` によるPython管理
- `uv`

PowerShellで `uv` を入れます。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後はPowerShellを開き直してください。

### 2. Python環境作成

```powershell
uv venv --python 3.12
uv pip install -e ".[onnxruntime,openvino,video,dashboard,dev]"
```

必要なら仮想環境をactivateします。

```powershell
.venv\Scripts\Activate.ps1
```

activate後は次のように実行できます。

```powershell
trt-profiler eval -c examples\squeezenet\config.yaml
```

activateしない場合は次のように直接実行できます。

```powershell
.venv\Scripts\trt-profiler.exe eval -c examples\squeezenet\config.yaml
```

### 3. WindowsでTensorRTを試す場合

Windows版TensorRTを試す場合は、NVIDIA TensorRT for Windowsを導入し、`trtexec.exe` と必要な
DLLのパスが見える状態にしてください。

config例:

```yaml
builder:
  config:
    trtexec: C:\path\to\TensorRT\bin\trtexec.exe
```

ただし、このrepositoryで十分に検証しているTensorRT実行経路はWSL2/Linuxです。

## SqueezeNet サンプル

公開ONNXモデルとサンプル入力を準備します。

```bash
.venv/bin/python examples/squeezenet/prepare_assets.py
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe examples\squeezenet\prepare_assets.py
```

生成されるファイル:

```text
examples/squeezenet/assets/model.onnx
examples/squeezenet/inputs/sample_0001.npz
examples/squeezenet/images/sample_0001.png
examples/squeezenet/videos/sample.mp4
```

## 実行例

### ONNX Runtime smoke

```bash
.venv/bin/trt-profiler eval -c examples/squeezenet/config.yaml
```

出力:

```text
examples/squeezenet/reports/report.json
```

### 後処理込みの分類metric

SqueezeNetのraw logitsを `SoftmaxPostprocessor` で確率に変換し、post stageで
`ClassificationConsistencyMetric` を評価します。

```bash
.venv/bin/trt-profiler eval -c examples/squeezenet/config_postprocess.yaml
```

出力:

```text
examples/squeezenet/reports_postprocess/report.json
examples/squeezenet/reports_postprocess/dashboard.html
examples/squeezenet/reports_postprocess/csv/
```

### ONNX Runtime vs OpenVINO

```bash
.venv/bin/trt-profiler eval -c examples/squeezenet/config_ort_openvino.yaml
```

出力:

```text
examples/squeezenet/reports_ort_openvino/report.json
examples/squeezenet/reports_ort_openvino/dashboard.html
```

### 画像入力

```bash
.venv/bin/trt-profiler eval -c examples/squeezenet/config_image_ort_openvino.yaml
```

### 動画入力

```bash
.venv/bin/trt-profiler eval -c examples/squeezenet/config_video_ort_openvino.yaml
```

### ONNX Runtime CPU vs TensorRT FP32

TensorRT v11と `trtexec` が必要です。

```bash
.venv/bin/trt-profiler eval -c examples/squeezenet/config_ort_trt_fp32.yaml
```

### TensorRT FP32 vs TensorRT FP16

TensorRT v11と `trtexec` が必要です。

```bash
.venv/bin/trt-profiler eval -c examples/squeezenet/config_trt_fp32_fp16.yaml
```

RTX 2060上のSqueezeNetサンプル結果例:

```text
shape_match_rate: 1.0
max_abs_error: 0.0164399147
mean_abs_error: 0.0035541029
rmse: 0.0045844717
cosine_similarity: 0.9999991308
```

## 比較ペアの自動生成

通常はconfigの `common.comparisons` に比較ペアを明示します。

一度の実行でconfig内の全variantの組み合わせを評価したい場合は、`--all-combinations` を使います。
variant数がN個なら、比較数は `N * (N - 1) / 2` になります。

```bash
.venv/bin/trt-profiler eval \
  -c examples/squeezenet/config_full_variants.yaml \
  --all-combinations
```

reference variantだけを基準にして他variantすべてと比較したい場合は、`--reference-to-all` を使います。
`role: reference` のvariantが基準になります。

```bash
.venv/bin/trt-profiler eval \
  -c examples/squeezenet/config_full_variants.yaml \
  --reference-to-all
```

config側に固定で書きたい場合は、`common.comparison_mode` に次の値を指定できます。

```yaml
common:
  comparison_mode: all-pairs
```

指定できる値:

- `configured`: `common.comparisons` を使う既定動作
- `all-pairs`: すべてのvariant組み合わせを生成
- `reference-to-all`: `role: reference` のvariantから他variantへの比較を生成

## 既存reportからdashboardを生成

```bash
.venv/bin/trt-profiler dashboard \
  examples/squeezenet/reports_ort_openvino/report.json \
  -o examples/squeezenet/reports_ort_openvino/dashboard.html
```

生成されたHTMLをブラウザで開くと、Plotly dashboardとして確認できます。

dashboardは単一の静的HTMLです。Dashサーバを起動しなくても、ブラウザ内で次を切り替えられます。

- 比較ペア
- `raw` / `post` stage
- metric

表示内容:

- 全比較ペアのcomparison matrix
- metric別ランキング
- summary metricのbar chart
- sampleごとのmetric分布
- summary table
- failed case table

より複雑な絞り込み、複数report横断、画像/heatmap previewなどが必要になった場合は、同じ
`report.json` を入力としてDashサーバ版を追加する想定です。

## Dashサーバ版dashboard

より複雑な確認にはDashサーバ版を使えます。複数の `report.json` を同時に読み込み、ブラウザ上で
横断的にfilterできます。

インストール:

```bash
uv pip install -e ".[dashboard]"
```

単一report:

```bash
.venv/bin/trt-profiler dash examples/squeezenet/reports_trt/report.json \
  --host 127.0.0.1 \
  --port 8050
```

複数report:

```bash
.venv/bin/trt-profiler dash \
  examples/squeezenet/reports_ort_openvino/report.json \
  examples/squeezenet/reports_trt/report.json
```

Dash版でできること:

- 複数report横断
- report / comparison / stage / metric のDropdown filter
- `?` ボタンによるmetric help modal
- comparison matrix
- metric ranking
- per-sample分布
- summary / failed case DataTable
- failed caseまたはsample row選択時の詳細表示
- heatmap `.npy` preview
- source image preview
- source image + heatmap overlay

metric help modalでは、選択中metricについて次を確認できます。

- metricの目的
- 主なstatの意味
- 値の見方の目安
- raw metric / post metricを見るときの注意点

画像previewには `source_path` が必要です。新しいreportでは `Sample.source_path` がper-sample rowに保存されます。
heatmap previewには `FeatureMapDiffMetric` の `save_heatmaps: true` で出力される `heatmap_path` を使います。

## CSV出力

`CsvReporter` を使うと、dashboardやJSONと同じ `ReportData` からCSVを出力できます。

```yaml
common:
  report:
    reporters:
      - class: trt_profiler.report.JsonReporter
        config:
          path: reports/report.json
      - class: trt_profiler.report.PlotlyDashboardReporter
        config:
          path: reports/dashboard.html
      - class: trt_profiler.report.CsvReporter
        config:
          output_dir: reports/csv
```

出力例:

```text
reports/csv/metric_summary.csv
reports/csv/per_sample.csv
reports/csv/per_output.csv
reports/csv/per_layer.csv
reports/csv/failed_cases.csv
reports/csv/worst_cases.csv
```

必要なtableだけ出したい場合は `tables` を指定します。

```yaml
      - class: trt_profiler.report.CsvReporter
        config:
          output_dir: reports/csv
          tables:
            - metric_summary
            - per_sample
```

## Config概要

configは大きく次の構成です。

- `common.model`: 元ONNXモデルの情報
- `common.variants`: 比較対象backendとbuilder/runner
- `common.dataset`: 入力データソース
- `common.comparisons`: reference / targetの比較ペア
- `common.input_mapping`: 共通入力名からbackend入力名へのmapping
- `common.output_mapping`: backend出力名から共通出力名へのmapping
- `common.metrics`: raw / postprocess metric
- `common.report`: JSON / dashboard reporter
- `preprocess`: 前処理classとconfig
- `postprocessors`: 後処理classとconfig

前処理/後処理classはそれぞれ自分のconfigを解釈します。pipeline側は基底classの契約だけに依存するため、
モデル固有の処理を低依存で差し替えられます。

## 簡易config

`common` セクションを持つconfigは詳細configとしてそのまま使われます。
`common` がないconfigは簡易configとして読み込まれ、内部で詳細configへ展開されます。

```text
YAML
  -> trt_profiler.config.parser で EvaluationConfig にparse
  -> trt_profiler.config.validation で必須項目を検証
  -> trt_profiler.config.presets でvariant/metric/component presetを解決
  -> trt_profiler.config.builder で既存pipeline用の詳細configへ変換
```

`trt_profiler.config.loader.load_config()` がこの流れをまとめています。
既存互換のため `trt_profiler.config.simple.normalize_config()` も残していますが、新規実装では
`loader.py`、`schema.py`、`parser.py`、`presets.py`、`builder.py` の責務を分けて追える構成です。

最小例:

```yaml
model:
  name: squeezenet1_1
  path: examples/squeezenet/assets/model.onnx

input:
  type: npz
  path: examples/squeezenet/inputs
  input_name: input_tensor
  backend_name: data
  npz_key: data

outputs:
  logits: squeezenet0_flatten0_reshape0

variants:
  - ort_cpu
  - openvino_cpu

compare: reference-to-all

metrics:
  raw:
    - tensor_diff:
        outputs: [logits]

report:
  output_dir: reports/simple
  formats: [json, dashboard, csv]
```

対応しているvariant preset:

- `ort_cpu`
- `ort_cuda`
- `ort_trt`
- `openvino_cpu`
- `trt_fp32`
- `trt_fp16`

metric preset:

- `tensor_diff`
- `feature_diff`
- `classification_consistency`
- `classification_accuracy`
- `detection_consistency`
- `detection_accuracy`

標準preprocessを使う場合:

```yaml
preprocess:
  type: imagenet
  input_name: images
  size: [224, 224]
```

独自preprocessを使う場合:

```yaml
preprocess:
  class: my_project.preprocess.MyPreprocessor
  config:
    input_name: images
    resize: [640, 640]
    letterbox: true
```

`class` を指定した場合はpreset展開せず、そのまま詳細configへ渡します。`config` の中身は独自class側で解釈してください。

postprocessも同じ考え方です。

```yaml
postprocess:
  type: softmax
  logits_key: logits
  probs_key: probs
```

または独自class:

```yaml
postprocess:
  class: my_project.postprocess.MyPostprocessor
  config:
    score_threshold: 0.25
    nms_iou: 0.45
```

SqueezeNetサンプルの `config*.yaml` は簡易config形式で整理しています。
一覧は [examples/squeezenet/README.md](examples/squeezenet/README.md) を参照してください。

## デフォルトの前処理/後処理

### 前処理

`trt_profiler.preprocessors.NpzPreprocessor`

- NPZ loaderの `Sample.data` からnumpy配列を取り出します。
- `inputs` を省略すると、NPZ内の全配列をそのまま入力にします。
- `inputs` を指定すると、共通入力名、NPZ key、dtypeを制御できます。

```yaml
preprocess:
  class: trt_profiler.preprocessors.NpzPreprocessor
  config:
    inputs:
      - name: input_tensor
        npz_key: data
        dtype: float32
```

`trt_profiler.preprocessors.ImageNetPreprocessor`

- 画像path、または動画loaderから渡されたBGR frameをRGB画像として処理します。
- resize、`0..1` 正規化、mean/std normalize、NCHW/NHWC変換に対応します。

```yaml
preprocess:
  class: trt_profiler.preprocessors.ImageNetPreprocessor
  config:
    input_name: data
    size: [224, 224]
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    layout: NCHW
    dtype: float32
```

### 後処理

`trt_profiler.postprocessors.IdentityPostprocessor`

- raw outputをそのままpostprocess結果として返します。
- raw比較だけで十分な場合や、post metricの開発初期に使えます。

```yaml
postprocessors:
  - name: identity
    class: trt_profiler.postprocessors.IdentityPostprocessor
    config: {}
```

`trt_profiler.postprocessors.SoftmaxPostprocessor`

- raw logitsをsoftmaxして確率に変換します。
- 分類post metricと組み合わせるためのサンプル実装です。

```yaml
postprocessors:
  - name: softmax
    class: trt_profiler.postprocessors.SoftmaxPostprocessor
    config:
      logits_key: logits
      probs_key: probs
      axis: -1
      squeeze: true
```

## デフォルトの評価指標

### 実装済み

`trt_profiler.metrics.TensorDiffMetric`

後処理前のraw tensor比較、またはpostprocess後にnumpy配列を比較したい場合に使う汎用metricです。

出力される主な値:

- `shape_match`: sampleごとのshape一致
- `shape_match_rate`: shape一致率
- `max_abs_error`: 最大絶対誤差
- `mean_abs_error`: 平均絶対誤差
- `rmse`: root mean square error
- `max_rel_error`: 最大相対誤差
- `mean_rel_error`: 平均相対誤差
- `cosine_similarity`: cosine類似度
- `allclose_rate`: `np.isclose` の一致率
- `nan_count`: target出力内のNaN数
- `inf_count`: target出力内のInf数
- `p95_abs_error` など: `percentiles` で指定した絶対誤差percentile

config例:

```yaml
common:
  metrics:
    raw:
      - name: logits_tensor_diff
        class: trt_profiler.metrics.TensorDiffMetric
        config:
          outputs:
            - logits
          atol: 1.0e-5
          rtol: 1.0e-5
          relative_eps: 1.0e-12
          percentiles:
            - 95
            - 99
```

`trt_profiler.metrics.FeatureMapDiffMetric`

- CNNバックボーンなどのfeature map比較用metricです。
- `TensorDiffMetric` の統計に加えて、layerごとのcosine、channelwise cosine、worst sampleを出力します。
- `save_heatmaps: true` にすると、spatial mean absolute errorを `.npy` heatmapとして保存します。

```yaml
common:
  metrics:
    raw:
      - name: feature_diff
        class: trt_profiler.metrics.FeatureMapDiffMetric
        config:
          outputs:
            - layer1
            - layer2
          channel_axis: 1
          save_heatmaps: true
          heatmap_dir: artifacts/heatmaps
          percentiles:
            - 95
```

`trt_profiler.metrics.ClassificationConsistencyMetric`

- referenceとtargetのtop-k classが一致するかを比較します。
- `probs_key` にはclass score/probabilityが入ったkeyを指定します。
- top-k一致率、KL divergence、JS divergence、softmax後score差分、ranking一致率を出力します。

```yaml
common:
  metrics:
    post:
      - name: cls_consistency
        class: trt_profiler.metrics.ClassificationConsistencyMetric
        config:
          probs_key: probs
          topk: [1, 5]
          apply_softmax: false
```

`trt_profiler.metrics.ClassificationAccuracyMetric`

- sampleのlabelを使ってtop-k accuracyを計算します。
- labelは `Sample.label`、または `sample.metadata[label_key]` から読みます。

```yaml
common:
  metrics:
    post:
      - name: cls_accuracy
        class: trt_profiler.metrics.ClassificationAccuracyMetric
        config:
          probs_key: probs
          topk: [1, 5]
          label_key: label
```

`trt_profiler.metrics.DetectionConsistencyMetric`

- referenceとtargetの検出結果をIoUで対応付けます。
- box IoU、対応数、class一致率、confidence差分、NMS後のbox数差分を評価します。
- 検出結果はデフォルトで `boxes`, `scores`, `labels` keyを読みます。

```yaml
common:
  metrics:
    post:
      - name: det_consistency
        class: trt_profiler.metrics.DetectionConsistencyMetric
        config:
          boxes_key: boxes
          scores_key: scores
          labels_key: labels
          iou_threshold: 0.5
          class_aware: true
```

`trt_profiler.metrics.DetectionAccuracyMetric`

- 簡易AP/mAPを計算します。
- `ground_truth_source: annotations` では `Sample.annotations` をGTとして使います。
- `ground_truth_source: reference` ではreference出力をGT扱いしてtargetを評価します。

```yaml
common:
  metrics:
    post:
      - name: det_accuracy
        class: trt_profiler.metrics.DetectionAccuracyMetric
        config:
          boxes_key: boxes
          scores_key: scores
          labels_key: labels
          iou_thresholds: [0.5, 0.75]
          ground_truth_source: annotations
```

## 自前の前処理を実装する

前処理は `trt_profiler.core.types.Preprocessor` を継承して実装します。
入力は `Sample`、戻り値は `dict[str, np.ndarray]` です。

例として、画像を読み込み、モデル入力名 `images` で返す前処理を作ります。

```python
# my_project/preprocess.py
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from trt_profiler.core.types import Preprocessor, Sample, TensorDict


class MyImagePreprocessor(Preprocessor):
    """Custom image preprocessor.

    Config Keys
    -----------
    input_name : str, optional
        Output input tensor name. Defaults to ``"images"``.
    size : list[int], optional
        Resize size as ``[width, height]``. Defaults to ``[640, 640]``.
    """

    def __call__(self, sample: Sample) -> TensorDict:
        image_path = Path(sample.data)
        image = Image.open(image_path).convert("RGB")
        width, height = self.config.get("size", [640, 640])
        image = image.resize((int(width), int(height)))

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))[None, ...]

        input_name = str(self.config.get("input_name", "images"))
        return {input_name: array}
```

configではdotted pathで指定します。

```yaml
preprocess:
  class: my_project.preprocess.MyImagePreprocessor
  config:
    input_name: images
    size: [640, 640]
```

pipeline側は `Preprocessor` の契約だけを見ます。configの中身は自前class内で解釈してください。

## 自前の後処理を実装する

後処理は `trt_profiler.core.types.Postprocessor` を継承します。
入力はraw output tensor、戻り値はmetricが比較しやすい辞書です。

```python
# my_project/postprocess.py
from __future__ import annotations

import numpy as np

from trt_profiler.core.types import ComparableDict, Postprocessor, Sample, TensorDict


class MyClassificationPostprocessor(Postprocessor):
    """Convert logits to probabilities."""

    def __call__(self, outputs: TensorDict, sample: Sample | None = None) -> ComparableDict:
        logits_key = str(self.config.get("logits_key", "logits"))
        probs_key = str(self.config.get("probs_key", "probs"))

        logits = np.asarray(outputs[logits_key], dtype=np.float64)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)
        return {probs_key: probs}
```

config例:

```yaml
postprocessors:
  - name: cls_post
    class: my_project.postprocess.MyClassificationPostprocessor
    config:
      logits_key: logits
      probs_key: probs
```

複数のpostprocessorを指定した場合、戻り値のkeyが衝突するとエラーになります。

## 自前の評価指標を実装する

評価指標は `trt_profiler.core.types.Metric` を継承します。
`update()` はsampleごとのrecordを返し、`compute()` は集計recordを返します。

```python
# my_project/metrics.py
from __future__ import annotations

from typing import Any

import numpy as np

from trt_profiler.core.types import Metric, MetricSummaryRecord, Sample, SampleMetricRecord


class MaxScoreDiffMetric(Metric):
    """Compare max score difference."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, config=config)
        self._values: list[float] = []

    def update(
        self,
        reference: dict[str, Any],
        target: dict[str, Any],
        sample: Sample | None = None,
    ) -> list[SampleMetricRecord]:
        key = str(self.config.get("key", "probs"))
        diff = abs(float(np.max(target[key])) - float(np.max(reference[key])))
        self._values.append(diff)

        return [
            SampleMetricRecord(
                sample_id=sample.id if sample is not None else "",
                comparison="",
                stage="",
                metric=self.name,
                output=key,
                stat="max_score_diff",
                value=diff,
            )
        ]

    def compute(self) -> list[MetricSummaryRecord]:
        mean_value = float(np.mean(self._values)) if self._values else 0.0
        max_value = float(np.max(self._values)) if self._values else 0.0
        key = str(self.config.get("key", "probs"))

        return [
            MetricSummaryRecord(
                comparison="",
                stage="",
                metric=self.name,
                output=key,
                stat="max_score_diff_mean",
                value=mean_value,
            ),
            MetricSummaryRecord(
                comparison="",
                stage="",
                metric=self.name,
                output=key,
                stat="max_score_diff_max",
                value=max_value,
            ),
        ]
```

raw tensorに対して使う場合:

```yaml
common:
  metrics:
    raw:
      - name: max_score_diff
        class: my_project.metrics.MaxScoreDiffMetric
        config:
          key: logits
```

postprocess後の結果に対して使う場合:

```yaml
common:
  metrics:
    post:
      - name: max_score_diff
        class: my_project.metrics.MaxScoreDiffMetric
        config:
          key: probs
```

metric実装側では `comparison` と `stage` を空文字のまま返して構いません。
pipelineの `Evaluator` が実行中のcomparison名と `raw` / `post` stageを埋めます。

## トラブルシュート

### `trtexec was not found on PATH`

TensorRTのcommand-line toolを導入するか、TensorRT builder configで `trtexec` pathを明示してください。

### `error while loading shared libraries: libnvinfer.so...`

TensorRTの共有ライブラリが見えていません。SDKを使っている場合はlibrary directoryを追加してください。

```bash
export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
```

### `Unknown option: --fp16`

TensorRT v11の `trtexec` では起こり得ます。このrepositoryでは `--fp16` を使わず、
`TensorRTBuilder` がONNXをFP16化してからengine buildします。

### WSL2でGPUが見えない

WSL2内で `nvidia-smi` を実行してください。失敗する場合はWindows側のNVIDIA driver更新や
WSL2 GPU supportの設定を確認してください。

### Windows path と WSL2 path の違い

Windowsの `C:\...` とWSL2の `/mnt/c/...` は別表記です。Windows上で実行するconfigにはWindows path、
WSL2上で実行するconfigにはLinux pathを使ってください。
