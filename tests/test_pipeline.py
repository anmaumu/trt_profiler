from __future__ import annotations

from pathlib import Path

import numpy as np

from trt_profiler.core.pipeline import EvaluationPipeline


def test_pipeline_writes_json_report(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.savez(input_dir / "sample_0001.npz", input=np.array([1.0, 2.0], dtype=np.float32))

    report_dir = tmp_path / "reports"
    config = {
        "common": {
            "model": {
                "name": "dummy",
                "source_path": str(tmp_path / "model.onnx"),
                "format": "onnx",
            },
            "variants": [
                {
                    "name": "ref",
                    "backend": "identity",
                    "role": "reference",
                    "builder": {
                        "class": "trt_profiler.builders.OnnxPassthroughBuilder",
                        "config": {},
                    },
                    "runner": {
                        "class": "trt_profiler.runners.IdentityRunner",
                        "config": {"outputs": {"output": "actual_input"}},
                    },
                },
                {
                    "name": "target",
                    "backend": "identity",
                    "role": "target",
                    "builder": {
                        "class": "trt_profiler.builders.OnnxPassthroughBuilder",
                        "config": {},
                    },
                    "runner": {
                        "class": "trt_profiler.runners.IdentityRunner",
                        "config": {"outputs": {"output": "actual_input"}, "bias": 0.1},
                    },
                },
            ],
            "dataset": {"type": "npz_folder", "config": {"path": str(input_dir)}},
            "comparisons": [{"name": "ref_vs_target", "reference": "ref", "target": "target"}],
            "input_mapping": {
                "ref": {"actual_input": "image"},
                "target": {"actual_input": "image"},
            },
            "output_mapping": {
                "ref": {"logits": "output"},
                "target": {"logits": "output"},
            },
            "metrics": {
                "raw": [
                    {
                        "name": "tensor_diff",
                        "class": "trt_profiler.metrics.TensorDiffMetric",
                        "config": {"outputs": ["logits"]},
                    }
                ],
                "post": [],
            },
            "report": {"output_dir": str(report_dir)},
        },
        "preprocess": {
            "class": "trt_profiler.preprocessors.NpzPreprocessor",
            "config": {"inputs": [{"name": "image", "npz_key": "input"}]},
        },
        "postprocessors": [],
    }

    EvaluationPipeline(config).run()

    assert (report_dir / "report.json").exists()
