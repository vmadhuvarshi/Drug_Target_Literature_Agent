"""Report, chart, and CSV generation for benchmark runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRIC_LABELS = {
    "retrieval_precision": "Retrieval Precision",
    "retrieval_recall": "Retrieval Recall",
    "citation_accuracy": "Citation Accuracy",
    "factual_coverage": "Factual Coverage",
    "hallucination_rate": "Hallucination Rate",
}

RADAR_LABELS = {
    "retrieval_precision": "Retrieval\nPrecision",
    "retrieval_recall": "Retrieval\nRecall",
    "citation_accuracy": "Citation\nAccuracy",
    "factual_coverage": "Factual\nCoverage",
    "hallucination_control": "Hallucination\nControl",
}


def write_markdown_report(run_result: dict[str, Any], output_path: Path) -> None:
    """Write a markdown summary report for one benchmark run."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = run_result.get("results", [])
    summary = run_result.get("summary", {})
    overall = summary.get("overall_metrics", {})
    category_breakdown = summary.get("category_breakdown", {})
    latency = summary.get("latency", {})
    worst = _worst_questions(results, limit=5)

    lines: list[str] = []
    lines.append(f"# Evaluation Report: {run_result.get('run_id', 'unknown')}")
    lines.append("")
    lines.append(f"- Timestamp: `{run_result.get('timestamp', 'unknown')}`")
    lines.append(f"- Model: `{run_result.get('model', 'unknown')}`")
    lines.append(f"- Questions evaluated: `{len(results)}`")
    lines.append(f"- Sources: `{', '.join(run_result.get('enabled_sources', []))}`")
    lines.append("")

    lines.append("## Overall Scores")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|---|---:|")
    for key, label in METRIC_LABELS.items():
        lines.append(f"| {label} | {_format_score(overall.get(key))} |")
    lines.append(f"| Composite Reliability | {_format_score(summary.get('composite_reliability'))} |")
    lines.append("")
    lines.append(
        "Composite reliability averages retrieval precision, retrieval recall, "
        "citation accuracy, factual coverage, and one minus hallucination rate."
    )
    lines.append("")

    lines.append("## Per-Category Breakdown")
    lines.append("")
    lines.append(
        "| Category | n | Retrieval Precision | Retrieval Recall | Citation Accuracy | "
        "Factual Coverage | Hallucination Rate | Composite |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for category in sorted(category_breakdown):
        row = category_breakdown[category]
        lines.append(
            f"| {category} | {row.get('count', 0)} | "
            f"{_format_score(row.get('retrieval_precision'))} | "
            f"{_format_score(row.get('retrieval_recall'))} | "
            f"{_format_score(row.get('citation_accuracy'))} | "
            f"{_format_score(row.get('factual_coverage'))} | "
            f"{_format_score(row.get('hallucination_rate'))} | "
            f"{_format_score(row.get('composite_reliability'))} |"
        )
    lines.append("")

    lines.append("## Worst-Performing Questions")
    lines.append("")
    if worst:
        lines.append("| Rank | Question ID | Category | Composite | Primary Gap |")
        lines.append("|---:|---|---|---:|---|")
        for idx, item in enumerate(worst, start=1):
            lines.append(
                f"| {idx} | `{item.get('id')}` | {item.get('category')} | "
                f"{_format_score(item.get('composite_reliability'))} | "
                f"{_primary_gap(item)} |"
            )
    else:
        lines.append("No completed question records were available.")
    lines.append("")

    lines.append("## Latency Statistics")
    lines.append("")
    lines.append("| Phase | Mean (s) | Median (s) | P95 (s) | Min (s) | Max (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for phase in ("retrieval_time", "synthesis_time", "metrics_time", "total_time"):
        row = latency.get(phase, {})
        lines.append(
            f"| {phase.replace('_', ' ').title()} | "
            f"{_format_seconds(row.get('mean'))} | "
            f"{_format_seconds(row.get('median'))} | "
            f"{_format_seconds(row.get('p95'))} | "
            f"{_format_seconds(row.get('min'))} | "
            f"{_format_seconds(row.get('max'))} |"
        )
    lines.append("")

    artifacts = run_result.get("artifacts", {})
    if artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for name, value in artifacts.items():
            lines.append(f"- {name}: `{value}`")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv_export(run_result: dict[str, Any], output_path: Path) -> None:
    """Write one flat CSV row per benchmark question."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "timestamp",
        "question_id",
        "category",
        "difficulty",
        "status",
        "question",
        "retrieval_precision",
        "retrieval_recall",
        "citation_accuracy",
        "factual_coverage",
        "hallucination_rate",
        "composite_reliability",
        "retrieval_time",
        "synthesis_time",
        "metrics_time",
        "total_time",
        "retrieved_count",
        "selected_count",
        "source_counts",
        "tool_queries",
        "error",
        "synthesis",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in run_result.get("results", []):
            timings = result.get("timings", {})
            metrics = result.get("metrics", {})
            pipeline = result.get("pipeline", {})
            row = {
                "run_id": run_result.get("run_id"),
                "timestamp": run_result.get("timestamp"),
                "question_id": result.get("id"),
                "category": result.get("category"),
                "difficulty": result.get("difficulty"),
                "status": result.get("status"),
                "question": result.get("question"),
                "retrieval_precision": _metric_score(metrics, "retrieval_precision"),
                "retrieval_recall": _metric_score(metrics, "retrieval_recall"),
                "citation_accuracy": _metric_score(metrics, "citation_accuracy"),
                "factual_coverage": _metric_score(metrics, "factual_coverage"),
                "hallucination_rate": _metric_score(metrics, "hallucination_rate"),
                "composite_reliability": result.get("composite_reliability"),
                "retrieval_time": timings.get("retrieval_time"),
                "synthesis_time": timings.get("synthesis_time"),
                "metrics_time": timings.get("metrics_time"),
                "total_time": timings.get("total_time"),
                "retrieved_count": len(pipeline.get("retrieved_papers", [])),
                "selected_count": len(pipeline.get("selected_evidence", [])),
                "source_counts": json.dumps(pipeline.get("source_counts", {}), sort_keys=True),
                "tool_queries": json.dumps(pipeline.get("tool_queries", [])),
                "error": result.get("error") or pipeline.get("error"),
                "synthesis": pipeline.get("synthesis", ""),
            }
            writer.writerow(row)


def write_radar_chart(run_result: dict[str, Any], output_path: Path) -> None:
    """Write a matplotlib radar chart comparing normalized metric dimensions."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = run_result.get("summary", {})
    overall = summary.get("overall_metrics", {})
    values = [
        _score_or_zero(overall.get("retrieval_precision")),
        _score_or_zero(overall.get("retrieval_recall")),
        _score_or_zero(overall.get("citation_accuracy")),
        _score_or_zero(overall.get("factual_coverage")),
        max(0.0, 1.0 - _score_or_zero(overall.get("hallucination_rate"))),
    ]
    labels = list(RADAR_LABELS.values())

    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    values += values[:1]
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, label="Overall")
    ax.fill(angles, values, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title("Evaluation Performance by Metric Dimension", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _worst_questions(results: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = [
        result
        for result in results
        if result.get("composite_reliability") is not None
    ]
    ranked.sort(key=lambda item: item.get("composite_reliability", 0.0))
    return ranked[:limit]


def _primary_gap(result: dict[str, Any]) -> str:
    metrics = result.get("metrics", {})
    scored = {
        "Retrieval Precision": _metric_score(metrics, "retrieval_precision"),
        "Retrieval Recall": _metric_score(metrics, "retrieval_recall"),
        "Citation Accuracy": _metric_score(metrics, "citation_accuracy"),
        "Factual Coverage": _metric_score(metrics, "factual_coverage"),
        "Hallucination Control": (
            None
            if _metric_score(metrics, "hallucination_rate") is None
            else 1.0 - _metric_score(metrics, "hallucination_rate")
        ),
    }
    scored = {key: value for key, value in scored.items() if value is not None}
    if not scored:
        return "No scored metrics"
    key = min(scored, key=lambda item: scored[item])
    return f"{key} ({scored[key]:.2f})"


def _metric_score(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key, {}).get("score")
    return value if isinstance(value, (int, float)) else None


def _format_score(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.2f}"


def _format_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.2f}"


def _score_or_zero(value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def summarize_values(values: list[float]) -> dict[str, float | None]:
    """Return mean, median, p95, min, and max for a list of numeric values."""

    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}

    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, math.ceil(0.95 * len(sorted_values)) - 1)
    return {
        "mean": round(statistics.mean(sorted_values), 4),
        "median": round(statistics.median(sorted_values), 4),
        "p95": round(sorted_values[p95_index], 4),
        "min": round(sorted_values[0], 4),
        "max": round(sorted_values[-1], 4),
    }

