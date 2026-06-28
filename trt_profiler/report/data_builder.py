"""Report data table builder."""

from __future__ import annotations

from dataclasses import asdict

from trt_profiler.core.types import EvaluationResult, ReportData


class ReportDataBuilder:
    """Build report-ready tables from evaluation results."""

    def build(self, result: EvaluationResult) -> ReportData:
        """Convert an evaluation result into report data.

        Parameters
        ----------
        result
            Evaluation result emitted by the pipeline.

        Returns
        -------
        ReportData
            Report metadata, nested summary, flat tables, and artifact metadata.
        """

        return ReportData(
            metadata=result.metadata,
            summary=result.summary,
            tables={
                "metric_summary": _summary_rows(result.summary),
                "per_sample": [asdict(record) for record in result.per_sample],
                "per_output": [],
                "per_layer": [],
                "failed_cases": [
                    asdict(record) for record in result.per_sample if record.status == "fail"
                ],
                "worst_cases": [],
            },
            artifacts={"plots": [], "failed_case_files": []},
        )


def _summary_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, dict):
        return rows
    for comparison, stages in comparisons.items():
        if not isinstance(stages, dict):
            continue
        for stage, metrics in stages.items():
            if not isinstance(metrics, dict):
                continue
            for metric, outputs in metrics.items():
                if not isinstance(outputs, dict):
                    continue
                for output, stats in outputs.items():
                    if not isinstance(stats, dict):
                        continue
                    for stat, value in stats.items():
                        rows.append(
                            {
                                "comparison": comparison,
                                "stage": stage,
                                "metric": metric,
                                "output": output,
                                "stat": stat,
                                "value": value,
                            }
                        )
    return rows
