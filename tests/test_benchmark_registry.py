from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from learn_nethack.benchmark_registry import (
    build_benchmark_comparison_report,
    load_benchmark_registry,
    validate_benchmark_registry,
    write_benchmark_comparison_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_benchmark_registry_is_valid() -> None:
    registry = load_benchmark_registry(ROOT / "benchmarks/nethack_benchmarks.json")

    assert registry["captured_at"] == "2026-07-09"
    assert registry["competitive_definition"]["required_external_benchmark_ids"] == [
        "balrog-nethack-llm-leader-2026-02-24",
        "hihack-learned-agent-mean-score-2023",
    ]


def test_competitive_report_requires_internal_and_both_external_gates() -> None:
    registry = load_benchmark_registry(ROOT / "benchmarks/nethack_benchmarks.json")
    metrics = {
        "balrog_nethack_progress_percent": {
            "estimate": 2.2,
            "lower": 1.6,
            "upper": 2.8,
        },
        "mean_nle_score": {
            "estimate": 1700.0,
            "lower": 1600.0,
            "upper": 1800.0,
        },
    }

    report = build_benchmark_comparison_report(
        registry=registry,
        candidate_id="gemma-candidate",
        candidate_metrics=metrics,
        completed_protocols={"balrog-nethack-llm", "nle-challenge-full-episode"},
        internal_training_proof_passed=True,
    )

    assert report["competitive"]
    required = [row for row in report["comparisons"] if row["required_for_competitive"]]
    assert all(row["passed"] for row in required)


def test_protocol_mismatch_cannot_be_called_competitive() -> None:
    registry = load_benchmark_registry(ROOT / "benchmarks/nethack_benchmarks.json")

    report = build_benchmark_comparison_report(
        registry=registry,
        candidate_id="short-smoke",
        candidate_metrics={
            "mean_nle_score": {
                "estimate": 9999.0,
                "lower": 9999.0,
                "upper": 9999.0,
            }
        },
        completed_protocols={"short-80-step-smoke"},
        internal_training_proof_passed=True,
    )

    assert not report["competitive"]
    assert {
        row["status"]
        for row in report["comparisons"]
        if row["required_for_competitive"]
    } == {"protocol_not_run"}


def test_registry_rejects_required_reference_only_benchmark() -> None:
    registry = load_benchmark_registry(ROOT / "benchmarks/nethack_benchmarks.json")
    registry["benchmarks"][3]["required_for_competitive"] = True
    registry["competitive_definition"]["required_external_benchmark_ids"].append(
        registry["benchmarks"][3]["benchmark_id"]
    )

    try:
        validate_benchmark_registry(registry)
    except ValueError as exc:
        assert "reference-only" in str(exc)
    else:
        raise AssertionError("required reference-only benchmark must fail")


def test_benchmark_report_writer_is_deterministic_json() -> None:
    report = {
        "schema_version": "learn-nethack.benchmark-comparison.v1",
        "candidate_id": "candidate",
        "competitive": False,
    }
    with TemporaryDirectory() as tmp:
        path = write_benchmark_comparison_report(Path(tmp) / "report.json", report)
        written = json.loads(path.read_text(encoding="utf-8"))

    assert written == report
