"""Aggregate matched local world-model proof reports across training seeds."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


METRIC_PATHS = {
    "one_step_changed_f1": ("evaluation", "{variant}", "one_step", "changed_cell_f1"),
    "one_step_char_accuracy": (
        "evaluation",
        "{variant}",
        "one_step",
        "full_frame_char_accuracy",
    ),
    "next_1_changed_f1": (
        "evaluation",
        "{variant}",
        "rollouts",
        "next_1",
        "changed_cell_f1",
    ),
    "next_5_changed_f1": (
        "evaluation",
        "{variant}",
        "rollouts",
        "next_5",
        "changed_cell_f1",
    ),
    "next_10_changed_f1": (
        "evaluation",
        "{variant}",
        "rollouts",
        "next_10",
        "changed_cell_f1",
    ),
    "next_10_char_accuracy": (
        "evaluation",
        "{variant}",
        "rollouts",
        "next_10",
        "full_frame_char_accuracy",
    ),
    "action_mrr": (
        "evaluation",
        "{variant}",
        "action_ranking",
        "mean_reciprocal_rank",
    ),
    "action_top1": (
        "evaluation",
        "{variant}",
        "action_ranking",
        "top1_accuracy",
    ),
}


def aggregate_world_model_reports(
    *,
    report_paths: Iterable[str | Path],
    out_path: str | Path,
) -> dict[str, Any]:
    paths = [Path(path) for path in report_paths]
    if len(paths) < 2:
        raise ValueError("at least two world-model proof reports are required")
    reports = [_read_json(path) for path in paths]
    contracts = [_read_json(Path(report["contract_path"])) for report in reports]
    _validate_reports(reports, contracts)

    variants: dict[str, Any] = {}
    for variant in ("deterministic", "diffusion"):
        variants[variant] = {
            metric: _summary(
                [
                    float(_nested(report, _variant_path(path, variant)))
                    for report in reports
                ]
            )
            for metric, path in METRIC_PATHS.items()
        }
    deltas = {
        metric: _summary(
            [
                float(_nested(report, _variant_path(path, "diffusion")))
                - float(_nested(report, _variant_path(path, "deterministic")))
                for report in reports
            ]
        )
        for metric, path in METRIC_PATHS.items()
    }
    per_run = [
        {
            "run_id": report["run_id"],
            "train_seed": int(contract["train_config"]["seed"]),
            "verdict": report["verdict"],
            "wandb": report["wandb"],
            "next_10_changed_f1_delta": report["comparison"][
                "next_10_changed_f1_delta"
            ],
            "next_10_macro_f1_delta": report["comparison"][
                "next_10_changed_f1_paired_bootstrap"
            ]["mean_difference"],
            "action_mrr_delta": report["comparison"]["action_ranking_mrr_delta"],
            "one_step_char_accuracy_delta": report["comparison"][
                "one_step_full_frame_char_accuracy_delta"
            ],
        }
        for report, contract in zip(reports, contracts, strict=True)
    ]
    action_losses = sum(item["action_mrr_delta"] <= 0.0 for item in per_run)
    supported_runs = sum(item["verdict"] == "supported" for item in per_run)
    aggregate = {
        "schema_version": "learn-nethack.local-world-model-aggregate.v1",
        "verdict": ("supported" if supported_runs == len(per_run) else "not_supported"),
        "run_count": len(per_run),
        "supported_run_count": supported_runs,
        "dataset_sha256": _dataset_sha256(contracts[0]),
        "eval_seed": int(contracts[-1]["eval_config"]["seed"]),
        "matched_parameter_count": int(contracts[0]["matched_parameter_count"]),
        "variants": variants,
        "diffusion_minus_deterministic": deltas,
        "per_run": per_run,
        "failure_summary": {
            "action_mrr_nonpositive_delta_runs": action_losses,
            "diffusion_next_10_f1_range": (
                variants["diffusion"]["next_10_changed_f1"]["max"]
                - variants["diffusion"]["next_10_changed_f1"]["min"]
            ),
            "all_runs_passed_pre_registered_gate": supported_runs == len(per_run),
        },
        "report_paths": [str(path) for path in paths],
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    return aggregate


def _validate_reports(reports: list[dict], contracts: list[dict]) -> None:
    if any(
        report.get("schema_version") != "learn-nethack.local-world-model-proof.v1"
        for report in reports
    ):
        raise ValueError("all inputs must be local world-model proof reports")
    dataset_identities = {_dataset_sha256(contract) for contract in contracts}
    if len(dataset_identities) != 1:
        raise ValueError("world-model reports use different datasets")
    parameter_counts = {contract["matched_parameter_count"] for contract in contracts}
    if len(parameter_counts) != 1:
        raise ValueError("world-model reports use different parameter counts")
    model_configs = {
        json.dumps(contract["model_config"], sort_keys=True) for contract in contracts
    }
    if len(model_configs) != 1:
        raise ValueError("world-model reports use different model configurations")
    eval_seeds = {contract["eval_config"]["seed"] for contract in contracts}
    if len(eval_seeds) != 1:
        raise ValueError("world-model reports use different evaluation seeds")


def _dataset_sha256(contract: dict[str, Any]) -> str:
    value = contract.get("dataset_sha256")
    if value:
        return str(value)
    manifest = _read_json(Path(contract["dataset_manifest_path"]))
    return str(manifest["dataset_sha256"])


def _summary(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "values": values,
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _variant_path(path: tuple[str, ...], variant: str) -> tuple[str, ...]:
    return tuple(part.format(variant=variant) for part in path)


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
