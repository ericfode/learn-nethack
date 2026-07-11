"""Versioned external benchmark contracts and comparison reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_benchmark_registry(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen NetHack benchmark registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_benchmark_registry(payload)
    return payload


def validate_benchmark_registry(registry: Mapping[str, Any]) -> None:
    if registry.get("schema_version") != "learn-nethack.benchmark-registry.v1":
        raise ValueError("unsupported NetHack benchmark registry schema")
    benchmarks = registry.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ValueError("benchmark registry must contain benchmarks")
    benchmark_ids: set[str] = set()
    required_ids: set[str] = set()
    for benchmark in benchmarks:
        if not isinstance(benchmark, Mapping):
            raise ValueError("benchmark entries must be objects")
        benchmark_id = str(benchmark.get("benchmark_id") or "")
        if not benchmark_id or benchmark_id in benchmark_ids:
            raise ValueError(f"invalid or duplicate benchmark_id: {benchmark_id!r}")
        benchmark_ids.add(benchmark_id)
        if not benchmark.get("protocol_id") or not benchmark.get("source"):
            raise ValueError(f"benchmark {benchmark_id} lacks protocol or source")
        if benchmark.get("required_for_competitive"):
            required_ids.add(benchmark_id)
            if benchmark.get("comparison") == "reference_only":
                raise ValueError(
                    f"required benchmark {benchmark_id} cannot be reference-only"
                )
            if not benchmark.get("candidate_metric"):
                raise ValueError(
                    f"required benchmark {benchmark_id} lacks candidate_metric"
                )
    definition = registry.get("competitive_definition")
    if not isinstance(definition, Mapping):
        raise ValueError("benchmark registry lacks competitive_definition")
    declared_required = set(definition.get("required_external_benchmark_ids") or [])
    if declared_required != required_ids:
        raise ValueError(
            "competitive_definition required IDs do not match benchmark entries"
        )


def build_benchmark_comparison_report(
    *,
    registry: Mapping[str, Any],
    candidate_id: str,
    candidate_metrics: Mapping[str, Mapping[str, float]],
    completed_protocols: set[str],
    internal_training_proof_passed: bool,
) -> dict[str, Any]:
    """Compare candidate confidence intervals only under matched protocols."""
    validate_benchmark_registry(registry)
    comparisons: list[dict[str, Any]] = []
    for benchmark in registry["benchmarks"]:
        benchmark_id = str(benchmark["benchmark_id"])
        protocol_id = str(benchmark["protocol_id"])
        comparison = str(benchmark["comparison"])
        required = bool(benchmark.get("required_for_competitive"))
        candidate_metric = benchmark.get("candidate_metric")
        result: dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "lane": benchmark["lane"],
            "protocol_id": protocol_id,
            "required_for_competitive": required,
            "source": benchmark["source"],
        }
        if comparison == "reference_only":
            result.update({"status": "reference_only", "passed": None})
        elif protocol_id not in completed_protocols:
            result.update({"status": "protocol_not_run", "passed": False})
        elif (
            not isinstance(candidate_metric, str)
            or candidate_metric not in candidate_metrics
        ):
            result.update({"status": "candidate_metric_missing", "passed": False})
        else:
            metric = candidate_metrics[candidate_metric]
            lower = metric.get("lower")
            estimate = metric.get("estimate")
            if not isinstance(lower, int | float) or not isinstance(
                estimate, int | float
            ):
                result.update({"status": "candidate_interval_missing", "passed": False})
            else:
                threshold = _benchmark_threshold(benchmark)
                passed = float(lower) >= threshold
                result.update(
                    {
                        "status": "passed" if passed else "failed",
                        "passed": passed,
                        "candidate_metric": candidate_metric,
                        "candidate": {
                            "estimate": float(estimate),
                            "lower": float(lower),
                            "upper": metric.get("upper"),
                        },
                        "threshold": threshold,
                        "comparison": comparison,
                    }
                )
        comparisons.append(result)

    required_results = [
        result for result in comparisons if result["required_for_competitive"]
    ]
    competitive = (
        internal_training_proof_passed
        and bool(required_results)
        and all(result.get("passed") is True for result in required_results)
    )
    return {
        "schema_version": "learn-nethack.benchmark-comparison.v1",
        "candidate_id": candidate_id,
        "registry_captured_at": registry["captured_at"],
        "internal_training_proof_passed": internal_training_proof_passed,
        "competitive": competitive,
        "comparisons": comparisons,
    }


def write_benchmark_comparison_report(
    path: str | Path,
    report: Mapping[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return target


def _benchmark_threshold(benchmark: Mapping[str, Any]) -> float:
    target = benchmark.get("target")
    if not isinstance(target, Mapping):
        raise ValueError(f"benchmark {benchmark.get('benchmark_id')} lacks target")
    comparison = benchmark.get("comparison")
    if comparison == "candidate_lower_at_least_target_lower":
        value = target.get("lower")
    elif comparison == "candidate_lower_at_least_target_estimate":
        value = target.get("estimate")
    else:
        raise ValueError(f"unsupported benchmark comparison: {comparison!r}")
    if not isinstance(value, int | float):
        raise ValueError(f"benchmark target is not numeric: {value!r}")
    return float(value)
