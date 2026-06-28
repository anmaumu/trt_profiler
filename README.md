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
  design.md        設計仕様とclass構成
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

## 既存reportからdashboardを生成

```bash
.venv/bin/trt-profiler dashboard \
  examples/squeezenet/reports_ort_openvino/report.json \
  -o examples/squeezenet/reports_ort_openvino/dashboard.html
```

生成されたHTMLをブラウザで開くと、Plotly dashboardとして確認できます。

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
