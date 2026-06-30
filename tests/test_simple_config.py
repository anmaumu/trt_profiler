from __future__ import annotations

from pathlib import Path

import yaml

from trt_profiler.config.loader import load_config
from trt_profiler.config.simple import normalize_config


def test_normalize_simple_config_expands_to_full_config() -> None:
    config = normalize_config(
        {
            "model": {"name": "dummy", "path": "model.onnx"},
            "input": {
                "type": "npz",
                "path": "inputs",
                "input_name": "input_tensor",
                "backend_name": "data",
                "npz_key": "data",
            },
            "outputs": {"logits": "output"},
            "variants": ["ort_cpu", {"name": "openvino_cpu", "preset": "openvino_cpu"}],
            "compare": "reference-to-all",
            "metrics": {
                "raw": [{"tensor_diff": {"outputs": ["logits"]}}],
                "post": [{"classification_consistency": {"probs_key": "probs"}}],
            },
            "postprocess": {"type": "softmax", "logits_key": "logits"},
            "report": {"output_dir": "reports", "formats": ["json", "dashboard", "csv"]},
        }
    )

    assert "common" in config
    assert config["common"]["variants"][0]["runner"]["config"]["providers"] == [
        "CPUExecutionProvider"
    ]
    assert config["common"]["comparison_mode"] == "reference-to-all"
    assert config["common"]["input_mapping"]["ort_cpu"] == {"data": "input_tensor"}
    assert config["common"]["metrics"]["raw"][0]["class"] == "trt_profiler.metrics.TensorDiffMetric"
    assert (
        config["postprocessors"][0]["class"] == "trt_profiler.postprocessors.SoftmaxPostprocessor"
    )
    assert config["common"]["report"]["reporters"][2]["class"] == "trt_profiler.report.CsvReporter"


def test_load_config_normalizes_simple_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  name: dummy
  path: model.onnx
input:
  type: npz
  path: inputs
outputs:
  logits: output
variants:
  - ort_cpu
  - openvino_cpu
metrics:
  raw:
    - tensor_diff:
        outputs: [logits]
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["common"]["model"]["name"] == "dummy"
    assert len(config["common"]["variants"]) == 2


def test_squeezenet_example_configs_are_concise_and_loadable() -> None:
    config_paths = sorted(Path("examples/squeezenet").glob("config*.yaml"))

    assert config_paths
    for config_path in config_paths:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert "common" not in raw, f"{config_path} should use concise config"

        config = load_config(config_path)
        assert "common" in config
        assert config["common"]["model"]["name"] == "squeezenet1_1"
