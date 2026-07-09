"""Mandatory W&B mirroring for local world-model proof runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def start_wandb_run(
    *,
    run_id: str,
    project: str,
    mode: str,
    target: Path,
    config: dict[str, Any],
):
    import wandb

    wandb_dir = target / ".wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        name=run_id,
        job_type="local-world-model-proof",
        mode=mode,
        dir=str(wandb_dir),
        config=config,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    return run


def progress_callback(
    *,
    variant: str,
    progress_path: Path,
    wandb_run,
    step_offset: int,
):
    def callback(metrics: dict[str, float | int], step: int) -> None:
        global_step = step_offset + step
        event = {
            "variant": variant,
            "variant_step": step,
            "global_step": global_step,
            **metrics,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        wandb_run.log(
            {f"{variant}/{name}": value for name, value in metrics.items()},
            step=global_step,
        )

    return callback


def wandb_identity(run, *, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "run_id": str(run.id),
        "run_name": str(run.name),
        "project": str(run.project),
        "entity": None if run.entity is None else str(run.entity),
        "url": getattr(run, "url", None),
        "offline_directory": getattr(run, "dir", None) if mode == "offline" else None,
    }


def log_wandb_artifact(
    *,
    wandb_run,
    run_id: str,
    target: Path,
    dataset_manifest_path: str | Path,
) -> None:
    import wandb

    artifact = wandb.Artifact(f"{run_id}-proof", type="world-model-proof")
    for path in (
        target / "run_contract.json",
        target / "report.json",
        target / "wandb_run.json",
        target / "training_progress.jsonl",
        target / "paired_next_10_changed_f1.npz",
        target / "watch" / "events.jsonl",
        target / "watch" / "index.html",
        Path(dataset_manifest_path),
    ):
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    for variant in ("deterministic", "diffusion"):
        checkpoint = target / "checkpoints" / f"{variant}.pt"
        if checkpoint.exists():
            artifact.add_file(str(checkpoint), name=f"checkpoints/{checkpoint.name}")
    wandb_run.log_artifact(artifact)


def flatten_numeric_report(payload: dict[str, Any]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, bool):
            result[prefix] = int(value)
        elif isinstance(value, (int, float)):
            result[prefix] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}/{key}" if prefix else str(key))

    visit(payload, "proof")
    return result
