from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trt_profiler.core.pipeline import EvaluationPipeline, build_comparisons
from trt_profiler.core.types import BackendVariant


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


def test_pipeline_runs_postprocess_metrics(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    np.savez(input_dir / "sample_0001.npz", input=np.array([1.0, 2.0, 0.5], dtype=np.float32))

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
                        "config": {"outputs": {"logits": "actual_input"}},
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
                        "config": {"outputs": {"logits": "actual_input"}, "bias": 0.01},
                    },
                },
            ],
            "dataset": {"type": "npz_folder", "config": {"path": str(input_dir)}},
            "comparisons": [{"name": "ref_vs_target", "reference": "ref", "target": "target"}],
            "input_mapping": {
                "ref": {"actual_input": "input_tensor"},
                "target": {"actual_input": "input_tensor"},
            },
            "output_mapping": {
                "ref": {"logits": "logits"},
                "target": {"logits": "logits"},
            },
            "metrics": {
                "raw": [],
                "post": [
                    {
                        "name": "classification_consistency",
                        "class": "trt_profiler.metrics.ClassificationConsistencyMetric",
                        "config": {"probs_key": "probs", "topk": [1, 2]},
                    }
                ],
            },
            "report": {"output_dir": str(tmp_path / "reports")},
        },
        "preprocess": {
            "class": "trt_profiler.preprocessors.NpzPreprocessor",
            "config": {"inputs": [{"name": "input_tensor", "npz_key": "input"}]},
        },
        "postprocessors": [
            {
                "name": "softmax",
                "class": "trt_profiler.postprocessors.SoftmaxPostprocessor",
                "config": {"logits_key": "logits", "probs_key": "probs"},
            }
        ],
    }

    result = EvaluationPipeline(config).run()

    post_summary = result.summary["comparisons"]["ref_vs_target"]["post"]
    metric_summary = post_summary["classification_consistency"]["probs"]
    assert metric_summary["top1_match_rate"] == 1.0
    assert "kl_divergence" in metric_summary


def test_build_comparisons_all_pairs() -> None:
    variants = [
        _variant("ort_cpu", "reference"),
        _variant("openvino_cpu", "target"),
        _variant("trt_fp32", "target"),
    ]

    comparisons = build_comparisons(variants, [], mode="all-pairs")

    assert [(item.name, item.reference, item.target) for item in comparisons] == [
        ("ort_cpu_vs_openvino_cpu", "ort_cpu", "openvino_cpu"),
        ("ort_cpu_vs_trt_fp32", "ort_cpu", "trt_fp32"),
        ("openvino_cpu_vs_trt_fp32", "openvino_cpu", "trt_fp32"),
    ]


def test_build_comparisons_reference_to_all() -> None:
    variants = [
        _variant("ort_cpu", "reference"),
        _variant("openvino_cpu", "target"),
        _variant("trt_fp32", "target"),
    ]

    comparisons = build_comparisons(variants, [], mode="reference-to-all")

    assert [(item.name, item.reference, item.target) for item in comparisons] == [
        ("ort_cpu_vs_openvino_cpu", "ort_cpu", "openvino_cpu"),
        ("ort_cpu_vs_trt_fp32", "ort_cpu", "trt_fp32"),
    ]


def test_build_comparisons_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown comparison mode"):
        build_comparisons([_variant("ort_cpu", "reference")], [], mode="unexpected")


def _variant(name: str, role: str) -> BackendVariant:
    return BackendVariant(
        name=name,
        backend="identity",
        role=role,
        builder_class="trt_profiler.builders.OnnxPassthroughBuilder",
        runner_class="trt_profiler.runners.IdentityRunner",
    )
