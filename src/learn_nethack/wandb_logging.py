"""Small W&B helpers for local data-loop reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def resolve_local_wandb_mode(env: Mapping[str, str | None] | None = None) -> str:
    """Return online/offline mode without silently disabling W&B."""
    effective_env = os.environ if env is None else env
    requested = (effective_env.get("WANDB_MODE") or "online").lower()
    if requested == "offline":
        return "offline"
    if effective_env.get("WANDB_API_KEY"):
        return "online"
    raise RuntimeError(
        "WANDB_API_KEY is required for online W&B runs; set WANDB_MODE=offline "
        "for explicit local smokes."
    )


def discover_offline_wandb_runs(root: str | Path = ".") -> list[Path]:
    """Return local W&B offline run directories under the repo artifact roots."""
    root_path = Path(root)
    search_roots = (root_path / "artifacts", root_path / "wandb")
    run_dirs: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("offline-run-*"):
            if path.is_dir() and any(path.glob("run-*.wandb")):
                run_dirs.add(path)
    return sorted(run_dirs)


def build_wandb_visibility_report(
    *,
    root: str | Path = ".",
    env: Mapping[str, str | None] | None = None,
) -> dict[str, object]:
    """Summarize why local W&B runs may not be visible in the web UI."""
    effective_env = os.environ if env is None else env
    offline_runs = discover_offline_wandb_runs(root)
    offline_paths = [str(path) for path in offline_runs]
    api_key_configured = bool(effective_env.get("WANDB_API_KEY"))
    requested_mode = (effective_env.get("WANDB_MODE") or "online").lower()
    sync_command = (
        "uv run wandb sync " + " ".join(offline_paths) if offline_paths else None
    )

    if offline_paths and not api_key_configured:
        next_step = (
            f"Set WANDB_API_KEY or run `uv run wandb login`, then run `{sync_command}`."
        )
    elif offline_paths:
        next_step = f"Run `{sync_command}` to upload the local offline runs."
    elif requested_mode == "offline":
        next_step = "WANDB_MODE=offline is set, but no local offline runs were found."
    elif api_key_configured:
        next_step = "No local offline runs were found; new runs should log online."
    else:
        next_step = (
            "No local offline runs were found and no WANDB_API_KEY is configured; "
            "online W&B runs cannot be created from this shell."
        )

    return {
        "schema_version": "learn-nethack.wandb-visibility.v1",
        "root": str(Path(root)),
        "requested_mode": requested_mode,
        "api_key_configured": api_key_configured,
        "offline_run_count": len(offline_paths),
        "offline_run_paths": offline_paths,
        "sync_command": sync_command,
        "recommended_next_step": next_step,
    }


def log_sft_build_to_wandb(
    *,
    output_dir: str | Path,
    metrics: Mapping[str, int | float],
    config: Mapping[str, object],
    env: Mapping[str, str | None] | None = None,
    project: str = "learn-nethack",
    run_name: str | None = None,
) -> str:
    """Log an SFT data-build report to W&B and return the resolved mode."""
    mode = resolve_local_wandb_mode(env)
    target = Path(output_dir)
    restored_env = _setdefault_local_wandb_storage(target)
    try:
        import wandb
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency is required in pyproject.
        _restore_env(restored_env)
        raise RuntimeError("wandb is required for SFT data-build logging") from exc

    init_kwargs: dict[str, object] = {
        "project": project,
        "name": run_name,
        "job_type": "sft-data-build",
        "mode": mode,
        "config": dict(config),
    }
    settings_factory = getattr(wandb, "Settings", None)
    if settings_factory is not None:
        init_kwargs["settings"] = settings_factory(
            x_disable_machine_info=True,
            x_disable_stats=True,
        )
    try:
        run = wandb.init(**init_kwargs)
        artifact = wandb.Artifact(
            name=f"sft-data-build-{target.name}",
            type="dataset",
        )
        for filename in (
            "manifest.json",
            "rejection_report.json",
            "split_manifest.json",
            "action_manifest.json",
            "sample_rows.jsonl",
        ):
            path = target / filename
            if path.exists():
                artifact.add_file(str(path))

        run.log({f"sft_data/{key}": value for key, value in sorted(metrics.items())})
        run.log_artifact(artifact)
        run.finish()
        return mode
    finally:
        _restore_env(restored_env)


def log_pseudo_label_audit_to_wandb(
    *,
    report_path: str | Path,
    report: Mapping[str, Any],
    config: Mapping[str, object],
    env: Mapping[str, str | None] | None = None,
    project: str = "learn-nethack",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Mirror a local pseudo-label audit report to a mandatory W&B run."""
    mode = resolve_local_wandb_mode(env)
    target = Path(report_path)
    restored_env = _setdefault_local_wandb_storage(target.parent)
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - required in pyproject.
        _restore_env(restored_env)
        raise RuntimeError("wandb is required for pseudo-label audit logging") from exc

    try:
        init_kwargs: dict[str, object] = {
            "project": project,
            "name": run_name,
            "job_type": "pseudo-label-audit",
            "mode": mode,
            "config": dict(config),
        }
        settings_factory = getattr(wandb, "Settings", None)
        if settings_factory is not None:
            init_kwargs["settings"] = settings_factory(
                x_disable_machine_info=True,
                x_disable_stats=True,
            )
        run = wandb.init(**init_kwargs)
        run_path = getattr(run, "path", None)
        if isinstance(run_path, tuple):
            run_path = list(run_path)
        artifact_name = f"pseudo-label-audit-{target.parent.name}"
        wandb_report = {
            key: value
            for key, value in {
                "mode": mode,
                "project": project,
                "run_id": getattr(run, "id", None),
                "run_name": getattr(run, "name", None),
                "run_url": getattr(run, "url", None),
                "run_path": run_path,
                "artifact_name": artifact_name,
            }.items()
            if value is not None
        }
        enriched_report = {**dict(report), "wandb": wandb_report}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(enriched_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run.log(_pseudo_label_audit_metrics(enriched_report))
        artifact = wandb.Artifact(name=artifact_name, type="evaluation")
        artifact.add_file(str(target))
        run.log_artifact(artifact)
        run.finish()
        return wandb_report
    finally:
        _restore_env(restored_env)


def log_sft_integrity_to_wandb(
    *,
    report_path: str | Path,
    report: Mapping[str, Any],
    config: Mapping[str, object],
    env: Mapping[str, str | None] | None = None,
    project: str = "learn-nethack",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Mirror a local SFT integrity report to a mandatory W&B run."""
    mode = resolve_local_wandb_mode(env)
    target = Path(report_path)
    restored_env = _setdefault_local_wandb_storage(target.parent)
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - required in pyproject.
        _restore_env(restored_env)
        raise RuntimeError("wandb is required for SFT integrity logging") from exc

    try:
        init_kwargs: dict[str, object] = {
            "project": project,
            "name": run_name,
            "job_type": "sft-integrity-audit",
            "mode": mode,
            "config": dict(config),
        }
        settings_factory = getattr(wandb, "Settings", None)
        if settings_factory is not None:
            init_kwargs["settings"] = settings_factory(
                x_disable_machine_info=True,
                x_disable_stats=True,
            )
        run = wandb.init(**init_kwargs)
        run_path = getattr(run, "path", None)
        if isinstance(run_path, tuple):
            run_path = list(run_path)
        artifact_name = f"sft-integrity-{target.parent.name}"
        wandb_report = {
            key: value
            for key, value in {
                "mode": mode,
                "project": project,
                "run_id": getattr(run, "id", None),
                "run_name": getattr(run, "name", None),
                "run_url": getattr(run, "url", None),
                "run_path": run_path,
                "artifact_name": artifact_name,
            }.items()
            if value is not None
        }
        enriched_report = {**dict(report), "wandb": wandb_report}
        target.write_text(
            json.dumps(enriched_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run.log(_sft_integrity_metrics(enriched_report))
        artifact = wandb.Artifact(name=artifact_name, type="dataset-validation")
        artifact.add_file(str(target))
        run.log_artifact(artifact)
        run.finish()
        return wandb_report
    finally:
        _restore_env(restored_env)


def _pseudo_label_audit_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for section_name in ("counts", "rates"):
        section = report.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for key, value in section.items():
            if isinstance(value, bool):
                metrics[f"pseudo_label_audit/{section_name}/{key}"] = float(value)
            elif isinstance(value, int | float):
                metrics[f"pseudo_label_audit/{section_name}/{key}"] = float(value)
    promotion = report.get("promotion")
    if isinstance(promotion, Mapping):
        for key, value in promotion.items():
            if isinstance(value, bool):
                metrics[f"pseudo_label_audit/promotion/{key}"] = float(value)
            elif isinstance(value, int | float):
                metrics[f"pseudo_label_audit/promotion/{key}"] = float(value)
    return metrics


def _sft_integrity_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    metrics = {"sft_integrity/passed": float(report.get("passed") is True)}
    counts = report.get("counts")
    if isinstance(counts, Mapping):
        for key, value in counts.items():
            if isinstance(value, int | float):
                metrics[f"sft_integrity/counts/{key}"] = float(value)
    action_distribution = report.get("action_distribution")
    if isinstance(action_distribution, Mapping):
        for key in ("unique_action_count", "dominant_action_rate"):
            value = action_distribution.get(key)
            if isinstance(value, int | float):
                metrics[f"sft_integrity/actions/{key}"] = float(value)
    dynamics = report.get("dynamics_diagnostics")
    if isinstance(dynamics, Mapping):
        for key, value in dynamics.items():
            if isinstance(value, int | float):
                metrics[f"sft_integrity/dynamics/{key}"] = float(value)
    return metrics


def _setdefault_local_wandb_storage(target: Path) -> dict[str, str | None]:
    root = target / ".wandb"
    defaults = {
        "WANDB_DIR": root / "runs",
        "WANDB_DATA_DIR": root / "data",
        "WANDB_CONFIG_DIR": root / "config",
        "WANDB_ARTIFACT_DIR": root / "artifacts",
        "WANDB_CACHE_DIR": root / "cache",
    }
    restored: dict[str, str | None] = {}
    for key, value in defaults.items():
        restored[key] = os.environ.get(key)
        effective = Path(os.environ.setdefault(key, str(value)))
        effective.mkdir(parents=True, exist_ok=True)
    return restored


def _restore_env(restored: Mapping[str, str | None]) -> None:
    for key, value in restored.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
