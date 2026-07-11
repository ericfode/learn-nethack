"""W&B and local-ledger lifecycle helpers for watch evaluations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any


CommitVolume = Callable[[str], bool]


def initialize_watch_wandb_run(
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    mode: str,
    job_type: str,
    commit_volume: CommitVolume,
) -> tuple[Any, Any, dict[str, Any]]:
    """Establish W&B and a running local ledger before model or NLE work."""
    import wandb

    try:
        run = wandb.init(
            project="learn-nethack",
            name=str(contract["run_id"]),
            job_type=job_type,
            mode=mode,
            config=dict(contract),
            dir=str(contract["artifacts"]["root"]),
        )
    except Exception as exc:
        failure_report = watch_failure_report(
            contract=contract,
            contract_path=contract_path,
            mode=mode,
            stage="wandb_init",
            error=exc,
        )
        _write_json(str(contract["artifacts"]["report"]), failure_report)
        _commit_watch_volumes(commit_volume)
        raise

    wandb_report = _wandb_run_report(run, mode)
    running_report = {
        "schema_version": "learn-nethack.watch-execution.v1",
        "run_id": str(contract["run_id"]),
        "status": "running",
        "contract_path": str(contract_path),
        "wandb_mode": mode,
        "wandb": wandb_report,
    }
    _write_json(str(contract["artifacts"]["report"]), running_report)
    _commit_watch_volumes(commit_volume)
    return wandb, run, wandb_report


def watch_failure_report(
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    mode: str,
    stage: str,
    error: Exception,
    wandb_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": "learn-nethack.watch-execution.v1",
        "run_id": str(contract["run_id"]),
        "status": "failed",
        "failure_stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "contract_path": str(contract_path),
        "wandb_mode": mode,
    }
    if wandb_report:
        report["wandb"] = dict(wandb_report)
    return report


def record_watch_failure(
    *,
    wandb: Any,
    run: Any,
    contract: Mapping[str, Any],
    contract_path: Path,
    mode: str,
    stage: str,
    error: Exception,
    wandb_report: Mapping[str, Any],
    commit_volume: CommitVolume,
) -> None:
    """Best-effort mirror of a failed watch run without masking its exception."""
    failure_report = watch_failure_report(
        contract=contract,
        contract_path=contract_path,
        mode=mode,
        stage=stage,
        error=error,
        wandb_report=wandb_report,
    )
    report_path = _write_json(
        str(contract["artifacts"]["report"]),
        failure_report,
    )
    reporting_errors: list[str] = []
    try:
        run.log({"watch/failed": 1.0})
        artifact = wandb.Artifact(
            name=f"watch-failure-{contract['run_id']}",
            type="evaluation",
        )
        artifact.add_file(str(report_path))
        artifact.add_file(str(contract_path))
        run.log_artifact(artifact)
    except Exception as reporting_error:
        reporting_errors.append(f"{type(reporting_error).__name__}: {reporting_error}")
    try:
        run.finish(exit_code=1)
    except Exception as reporting_error:
        reporting_errors.append(f"{type(reporting_error).__name__}: {reporting_error}")
    if reporting_errors:
        failure_report["reporting_errors"] = reporting_errors
        _write_json(report_path, failure_report)
    _commit_watch_volumes(commit_volume)


def watch_sweep_wandb_replay_media(
    wandb: Any,
    seed_reports: Any,
) -> dict[str, Any]:
    """Build one W&B HTML media value per existing seed replay."""
    if not isinstance(seed_reports, Sequence) or isinstance(seed_reports, (str, bytes)):
        return {}
    media: dict[str, Any] = {}
    for seed_report in seed_reports:
        if not isinstance(seed_report, Mapping):
            continue
        viewer_path = seed_report.get("viewer_path")
        if not viewer_path or not Path(str(viewer_path)).exists():
            continue
        seed = seed_report.get("seed", "unknown")
        media[f"watch_sweep/replay_seed_{seed}"] = wandb.Html(
            str(viewer_path),
            inject=False,
        )
    return media


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _wandb_run_report(run: Any, mode: str) -> dict[str, Any]:
    path = getattr(run, "path", None)
    if isinstance(path, tuple):
        path = list(path)
    return {
        key: value
        for key, value in {
            "mode": mode,
            "run_id": getattr(run, "id", None),
            "run_name": getattr(run, "name", None),
            "run_url": getattr(run, "url", None),
            "run_path": path,
        }.items()
        if value is not None
    }


def _commit_watch_volumes(commit_volume: CommitVolume) -> None:
    commit_volume("/watch")
    commit_volume("/runs")
