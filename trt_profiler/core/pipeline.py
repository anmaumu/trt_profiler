from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from trt_profiler.core.factory import (
    build_artifact_builder,
    build_dataset,
    build_metric,
    build_postprocessors,
    build_preprocessor,
    build_reporter,
    build_runner,
    parse_variants,
)
from trt_profiler.core.types import (
    BackendVariant,
    Comparison,
    ConfigDict,
    EvaluationResult,
    ModelRunner,
    SourceModel,
)
from trt_profiler.evaluators.evaluator import Evaluator, nested_summary
from trt_profiler.mapping.input_mapper import InputMapper
from trt_profiler.mapping.output_mapper import OutputMapper
from trt_profiler.report.data_builder import ReportDataBuilder


class EvaluationPipeline:
    def __init__(self, config: ConfigDict) -> None:
        self.config = config
        common = config["common"]
        model = common["model"]
        self.source_model = SourceModel(
            path=Path(str(model["source_path"])),
            format=str(model["format"]),
        )
        self.variants = parse_variants(common["variants"])
        self.comparisons = [
            Comparison(
                name=str(item["name"]),
                reference=str(item["reference"]),
                target=str(item["target"]),
            )
            for item in common["comparisons"]
        ]
        self.dataset = build_dataset(common["dataset"])
        self.preprocessor = build_preprocessor(config["preprocess"])
        self.postprocessors = build_postprocessors(config.get("postprocessors", []))
        self.input_mapper = InputMapper(common.get("input_mapping", {}))
        self.output_mapper = OutputMapper(common.get("output_mapping", {}))
        self.raw_metric_configs = list(common.get("metrics", {}).get("raw", []))
        self.post_metric_configs = list(common.get("metrics", {}).get("post", []))
        self.reporter_configs = common.get("report", {}).get("reporters")
        self.report_config = common.get("report", {})

    def run(self) -> EvaluationResult:
        runners = self._prepare_runners()
        raw_records = []
        summary_records = []

        evaluators = {
            comparison.name: (
                Evaluator(
                    "raw",
                    [build_metric(deepcopy(item)) for item in self.raw_metric_configs],
                ),
                Evaluator(
                    "post",
                    [build_metric(deepcopy(item)) for item in self.post_metric_configs],
                ),
            )
            for comparison in self.comparisons
        }

        for sample in self.dataset:
            common_inputs = self.preprocessor(sample)
            raw_outputs = {}
            post_outputs = {}

            for name, runner in runners.items():
                runner_inputs = self.input_mapper.map(name, common_inputs)
                mapped_outputs = self.output_mapper.map(name, runner.infer(runner_inputs))
                raw_outputs[name] = mapped_outputs
                post_outputs[name] = self._postprocess(mapped_outputs, sample)

            for comparison in self.comparisons:
                raw_evaluator, post_evaluator = evaluators[comparison.name]
                raw_evaluator.update(
                    comparison.name,
                    raw_outputs[comparison.reference],
                    raw_outputs[comparison.target],
                    sample,
                )
                post_evaluator.update(
                    comparison.name,
                    post_outputs[comparison.reference],
                    post_outputs[comparison.target],
                    sample,
                )

        for comparison in self.comparisons:
            raw_evaluator, post_evaluator = evaluators[comparison.name]
            raw_records.extend(raw_evaluator.records)
            raw_records.extend(post_evaluator.records)
            summary_records.extend(raw_evaluator.compute(comparison.name))
            summary_records.extend(post_evaluator.compute(comparison.name))

        result = EvaluationResult(
            metadata={
                "model": self.config["common"]["model"],
                "variants": [variant.name for variant in self.variants],
                "comparisons": [comparison.name for comparison in self.comparisons],
            },
            summary=nested_summary(summary_records),
            per_sample=raw_records,
        )
        self._write_reports(result)
        return result

    def _prepare_runners(self) -> dict[str, ModelRunner]:
        variant_by_name = {variant.name: variant for variant in self.variants}
        required_names = {
            name
            for comparison in self.comparisons
            for name in (comparison.reference, comparison.target)
        }

        runners: dict[str, ModelRunner] = {}
        for name in required_names:
            variant = variant_by_name[name]
            runner = self._build_runner_for_variant(variant)
            runner.load()
            runners[name] = runner
        return runners

    def _build_runner_for_variant(self, variant: BackendVariant) -> ModelRunner:
        builder = build_artifact_builder(variant)
        artifact = builder.build(self.source_model)
        return build_runner(variant, artifact)

    def _postprocess(self, outputs: dict[str, Any], sample: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for postprocessor in self.postprocessors:
            update = postprocessor(outputs, sample)
            overlap = set(result).intersection(update)
            if overlap:
                raise KeyError(f"Postprocessor output key collision: {sorted(overlap)}")
            result.update(update)
        return result or dict(outputs)

    def _write_reports(self, result: EvaluationResult) -> None:
        report_data = ReportDataBuilder().build(result)
        if self.reporter_configs:
            reporters = [build_reporter(item) for item in self.reporter_configs]
        else:
            output_dir = Path(str(self.report_config.get("output_dir", ".")))
            reporters = [
                build_reporter(
                    {
                        "class": "trt_profiler.report.JsonReporter",
                        "config": {"path": str(output_dir / "report.json")},
                    }
                )
            ]
        for reporter in reporters:
            reporter.write(report_data)
