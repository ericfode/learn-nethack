"""Command-line entrypoint for the Gemma/NetHack pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from learn_nethack.action_manifest import (
    build_action_manifest_from_nle_actions,
    load_action_manifest,
)
from learn_nethack.compare_watch import (
    DEFAULT_NLE_CHARACTER,
    run_checkpoint_compare,
    run_checkpoint_compare_sweep,
    write_compare_watch_contract,
    write_compare_watch_sweep_contract,
)
from learn_nethack.dynamics_play import (
    DynamicsModelSpec,
    TransformerNextFramePredictor,
    parse_action_id_list,
    read_ground_truth_frames_from_next_frame_rows,
    read_initial_frame,
    run_interactive_dynamics_session,
    run_scripted_dynamics_session,
    write_dynamics_play_contract,
)
from learn_nethack.eval_readiness import (
    build_eval_status_report,
    check_modal_eval_markers,
    write_eval_status_report,
)
from learn_nethack.full_build_readiness import (
    build_full_build_readiness_report,
    check_modal_full_build_markers,
    write_full_build_readiness_report,
)
from learn_nethack.modal_config import (
    DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
    DEFAULT_WATCH_PROOF_SEEDS,
    sft_existing_dataset_followup_commands,
)
from learn_nethack.nld_decode import iter_nld_ttyrec_batches, normalize_decoded_batch
from learn_nethack.nld_index import build_altorg_index
from learn_nethack.nld_metadata import inspect_nld_db
from learn_nethack.nld_metadata import (
    order_gameids_for_split_role_coverage,
    read_game_metadata,
    read_gameids,
)
from learn_nethack.pseudo_label_audit import (
    build_pseudo_label_audit_report,
    stratified_audit_gameids,
    write_pseudo_label_audit_report,
)
from learn_nethack.rollout_preferences import write_rollout_preference_jsonl
from learn_nethack.sft_build import SplitRowLimits, build_sft_from_decoded_batches
from learn_nethack.sft_eval import (
    build_score_to_beat_report,
    build_training_proof_gate_report,
)
from learn_nethack.sft_fixture import build_sft_overfit_fixture
from learn_nethack.sft_integrity import audit_sft_dataset, write_sft_integrity_report
from learn_nethack.sft_train import SftTrainConfig
from learn_nethack.status_dashboard import (
    DEFAULT_BASELINE_EVAL_RUN_ID,
    DEFAULT_FULL_BUILD_RUN_ID,
    write_status_dashboard,
)
from learn_nethack.wandb_logging import build_wandb_visibility_report
from learn_nethack.wandb_logging import (
    log_pseudo_label_audit_to_wandb,
    log_sft_build_to_wandb,
    log_sft_integrity_to_wandb,
)


app = typer.Typer(help="Gemma/NetHack pipeline commands.")
data_app = typer.Typer(help="Inspect and build local NLD datasets.")
sft_app = typer.Typer(help="Train and evaluate supervised fine-tuning runs.")
wandb_app = typer.Typer(help="Inspect local W&B visibility state.")
watch_app = typer.Typer(help="Run and inspect watchable NetHack rollouts.")
play_app = typer.Typer(help="Play learned NetHack models as local tools.")
dashboard_app = typer.Typer(help="Build local project dashboards.")
app.add_typer(data_app, name="data")
app.add_typer(sft_app, name="sft")
app.add_typer(wandb_app, name="wandb")
app.add_typer(watch_app, name="watch")
app.add_typer(play_app, name="play")
app.add_typer(dashboard_app, name="dashboard")


@app.callback()
def main() -> None:
    """Run learn-nethack commands."""


def resolve_build_row_limit(*, max_rows: int, full_dataset: bool) -> int | None:
    """Return the effective row cap for a data build."""
    if full_dataset:
        if max_rows != 1000:
            raise ValueError(
                "full dataset builds cannot combine with custom --max-rows"
            )
        return None
    return max_rows


def resolve_split_row_limits(
    *,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
    full_dataset: bool,
) -> SplitRowLimits | None:
    """Resolve an optional balanced capped-build contract."""
    if validation_rows == 0 and test_rows == 0:
        return None
    if full_dataset:
        raise ValueError("full dataset builds cannot use split row limits")
    if train_rows <= 0 or validation_rows <= 0 or test_rows <= 0:
        raise ValueError("train, validation, and test row limits must all be positive")
    return SplitRowLimits(
        train=train_rows,
        validation=validation_rows,
        test=test_rows,
    )


@data_app.command("inspect")
def inspect_data(
    db: Path = typer.Option(..., "--db", help="Path to local NLD ttyrecs.db."),
) -> None:
    """Inspect a local NLD metadata database."""
    report = inspect_nld_db(db)
    typer.echo(json.dumps(report.__dict__, indent=2, sort_keys=True))


@data_app.command("build-sft")
def build_sft(
    db: Path = typer.Option(..., "--db", help="Path to local NLD ttyrecs.db."),
    mode: str = typer.Option("single_frame", "--mode"),
    out: Path = typer.Option(..., "--out", help="Output artifact directory."),
    max_rows: int = typer.Option(1000, "--max-rows"),
    validation_rows: int = typer.Option(0, "--validation-rows"),
    test_rows: int = typer.Option(0, "--test-rows"),
    full_dataset: bool = typer.Option(
        False,
        "--full-dataset",
        help="Build every accepted policy row from the indexed dataset.",
    ),
    tasks: str = typer.Option("policy_action,next_frame", "--tasks"),
    action_manifest: Path | None = typer.Option(
        None,
        "--action-manifest",
        help="Path to action_manifest.json mapping NLD raw keys to NLE action ids.",
    ),
    batch_size: int = typer.Option(128, "--batch-size"),
    seq_length: int = typer.Option(32, "--seq-length"),
    token_budget: int = typer.Option(2_048, "--token-budget"),
    progress_interval: int = typer.Option(5_000, "--progress-interval"),
    seed: int = typer.Option(20260615, "--seed"),
    wandb_project: str = typer.Option("learn-nethack", "--wandb-project"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
) -> None:
    """Build local multi-task SFT rows from decoded NLD ttyrecs."""
    report = inspect_nld_db(db)
    try:
        effective_max_rows = resolve_build_row_limit(
            max_rows=max_rows,
            full_dataset=full_dataset,
        )
        split_row_limits = resolve_split_row_limits(
            train_rows=max_rows,
            validation_rows=validation_rows,
            test_rows=test_rows,
            full_dataset=full_dataset,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    task_names = tuple(part.strip() for part in tasks.split(",") if part.strip())
    if action_manifest is None:
        raise typer.BadParameter(
            "build-sft requires --action-manifest so raw NLD keypresses are "
            "mapped explicitly to active NLE action ids"
        )
    manifest = load_action_manifest(action_manifest)
    gameids = read_gameids(db)
    game_metadata = read_game_metadata(db)
    if split_row_limits is None:
        decode_gameids = gameids
        game_order_strategy = "gameid_ascending"
    else:
        decode_gameids = order_gameids_for_split_role_coverage(
            gameids,
            game_metadata_by_id=game_metadata,
            seed=seed,
        )
        game_order_strategy = "split_role_round_robin_v1"
    progress_path = out / "sft_build_progress.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")

    def _write_progress(event: object) -> None:
        payload = dict(getattr(event, "__dict__", {}))
        line = json.dumps(payload, sort_keys=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        typer.echo(line, err=True)

    result = build_sft_from_decoded_batches(
        dataset_name=report.dataset_name,
        mode=mode,
        batches=iter_nld_ttyrec_batches(
            dataset_name=report.dataset_name,
            batch_size=batch_size,
            seq_length=seq_length,
            dbfilename=str(db),
            gameids=decode_gameids,
            shuffle=False,
            loop_forever=False,
        ),
        action_manifest=manifest,
        gameids=gameids,
        game_metadata_by_id=game_metadata,
        out_dir=out,
        max_rows=None if split_row_limits is not None else effective_max_rows,
        seed=seed,
        tasks=task_names,
        split_row_limits=split_row_limits,
        progress_callback=_write_progress,
        progress_interval=progress_interval,
        game_order_strategy=game_order_strategy,
        token_budget=token_budget,
    )
    wandb_mode = log_sft_build_to_wandb(
        output_dir=out,
        metrics={
            "accepted_policy_rows": result.accepted_policy_rows,
            "accepted_next_frame_rows": result.accepted_next_frame_rows,
            "rejected_rows": result.rejected_rows,
        },
        config={
            "dataset_name": report.dataset_name,
            "db_path": str(db),
            "mode": mode,
            "tasks": list(task_names),
            "batch_size": batch_size,
            "seq_length": seq_length,
            "token_budget": token_budget,
            "max_rows": effective_max_rows,
            "split_row_limits": (
                split_row_limits.as_dict() if split_row_limits is not None else None
            ),
            "full_dataset": full_dataset,
            "seed": seed,
            "game_order_strategy": game_order_strategy,
            "action_manifest_path": str(action_manifest),
            "progress_path": str(progress_path),
            "progress_interval": progress_interval,
        },
        project=wandb_project,
        run_name=wandb_run_name,
    )
    typer.echo(
        json.dumps(
            {**result.__dict__, "wandb_mode": wandb_mode},
            indent=2,
            sort_keys=True,
        )
    )


@data_app.command("write-action-manifest")
def write_action_manifest(
    out: Path = typer.Option(..., "--out", help="Output action_manifest.json path."),
    env_id: str = typer.Option("NetHackChallenge-v0", "--env-id"),
) -> None:
    """Write an action manifest from the installed NLE action list."""
    manifest = build_action_manifest_from_nle_actions(env_id=env_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(out)
    typer.echo(
        json.dumps(
            {
                "env_id": manifest.env_id,
                "action_count": len(manifest.entries),
                "output_path": str(out),
            },
            indent=2,
            sort_keys=True,
        )
    )


@data_app.command("audit-pseudo-labels")
def audit_pseudo_labels(
    db: Path = typer.Option(..., "--db", help="Path to a labelled NLD ttyrecs.db."),
    action_manifest: Path = typer.Option(
        ...,
        "--action-manifest",
        help="Action manifest used to map true keypresses and pseudo labels.",
    ),
    out: Path = typer.Option(..., "--out", help="Output audit JSON report."),
    max_transitions: int = typer.Option(100_000, "--max-transitions"),
    max_games: int = typer.Option(96, "--max-games"),
    seed: int = typer.Option(20260709, "--seed"),
    batch_size: int = typer.Option(128, "--batch-size"),
    seq_length: int = typer.Option(32, "--seq-length"),
    wandb_project: str = typer.Option("learn-nethack", "--wandb-project"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
) -> None:
    """Audit frame-derived movement labels against true NLD keypresses."""
    db_report = inspect_nld_db(db)
    manifest = load_action_manifest(action_manifest)
    game_metadata = read_game_metadata(db)
    gameids = stratified_audit_gameids(
        game_metadata,
        max_games=max_games,
        seed=seed,
    )
    progress_path = out.parent / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")

    def _write_progress(event: dict[str, object]) -> None:
        line = json.dumps(event, sort_keys=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        typer.echo(line, err=True)

    transitions = (
        transition
        for batch in iter_nld_ttyrec_batches(
            dataset_name=db_report.dataset_name,
            batch_size=batch_size,
            seq_length=seq_length,
            dbfilename=str(db),
            gameids=gameids,
            shuffle=False,
            loop_forever=False,
        )
        for transition in normalize_decoded_batch(batch)
    )
    report = build_pseudo_label_audit_report(
        transitions=transitions,
        action_manifest=manifest,
        game_metadata_by_id=game_metadata,
        max_transitions=max_transitions,
        progress_callback=_write_progress,
    )
    report["input"] = {
        "db_path": str(db),
        "dataset_name": db_report.dataset_name,
        "action_manifest_path": str(action_manifest),
        "batch_size": batch_size,
        "seq_length": seq_length,
        "progress_path": str(progress_path),
        "selected_game_count": len(gameids),
        "selected_gameids": gameids,
        "seed": seed,
    }
    write_pseudo_label_audit_report(out, report)
    wandb_report = log_pseudo_label_audit_to_wandb(
        report_path=out,
        report=report,
        config={
            "dataset_name": db_report.dataset_name,
            "db_path": str(db),
            "action_manifest_path": str(action_manifest),
            "max_transitions": max_transitions,
            "max_games": max_games,
            "seed": seed,
            "batch_size": batch_size,
            "seq_length": seq_length,
        },
        project=wandb_project,
        run_name=wandb_run_name,
    )
    typer.echo(json.dumps({**report, "wandb": wandb_report}, indent=2, sort_keys=True))


@data_app.command("build-overfit-fixture")
def build_overfit_fixture(
    source_dir: Path = typer.Option(
        ...,
        "--source-dir",
        help="Existing SFT dataset containing task-specific train JSONL files.",
    ),
    out: Path = typer.Option(..., "--out", help="Output fixture directory."),
    rows_per_task: int = typer.Option(4, "--rows-per-task"),
) -> None:
    """Build a balanced tiny dataset for real-tokenizer overfit tests."""
    report = build_sft_overfit_fixture(
        source_dir=source_dir,
        output_dir=out,
        rows_per_task=rows_per_task,
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@data_app.command("audit-sft")
def audit_sft(
    dataset_dir: Path = typer.Option(
        ...,
        "--dataset-dir",
        help="Completed SFT dataset directory.",
    ),
    out: Path | None = typer.Option(None, "--out", help="Output integrity report."),
    expected_env_id: str = typer.Option(
        "NetHackChallenge-v0",
        "--expected-env-id",
    ),
    wandb_project: str = typer.Option("learn-nethack", "--wandb-project"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
) -> None:
    """Prove split, action-label, and dynamics-conditioning integrity."""
    report_path = out or dataset_dir / "integrity_report.json"
    report = audit_sft_dataset(dataset_dir, expected_env_id=expected_env_id)
    write_sft_integrity_report(report_path, report)
    wandb_report = log_sft_integrity_to_wandb(
        report_path=report_path,
        report=report,
        config={
            "dataset_dir": str(dataset_dir),
            "expected_env_id": expected_env_id,
        },
        project=wandb_project,
        run_name=wandb_run_name,
    )
    typer.echo(json.dumps({**report, "wandb": wandb_report}, indent=2, sort_keys=True))
    if not report["passed"]:
        raise typer.Exit(code=1)


@wandb_app.command("status")
def wandb_status(
    root: Path = typer.Option(Path("."), "--root", help="Repository root to inspect."),
) -> None:
    """Report local W&B offline runs and upload readiness."""
    report = build_wandb_visibility_report(root=root)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@wandb_app.command("log-pseudo-audit")
def log_existing_pseudo_audit(
    report_path: Path = typer.Option(
        ...,
        "--report",
        help="Existing local pseudo-label audit report to mirror.",
    ),
    wandb_project: str = typer.Option("learn-nethack", "--wandb-project"),
    wandb_run_name: str | None = typer.Option(None, "--wandb-run-name"),
) -> None:
    """Retry W&B mirroring without decoding the NLD corpus again."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "learn-nethack.pseudo-label-audit.v1":
        raise typer.BadParameter("report is not a pseudo-label audit v1 artifact")
    config = report.get("input")
    if not isinstance(config, dict):
        config = {}
    wandb_report = log_pseudo_label_audit_to_wandb(
        report_path=report_path,
        report=report,
        config=config,
        project=wandb_project,
        run_name=wandb_run_name,
    )
    typer.echo(json.dumps(wandb_report, indent=2, sort_keys=True))


@dashboard_app.command("build")
def build_dashboard(
    out: Path = typer.Option(
        Path("artifacts/status-dashboard"),
        "--out",
        help="Output directory for dashboard.json and index.html.",
    ),
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root containing artifacts/.",
    ),
    build_run_id: str = typer.Option(
        DEFAULT_FULL_BUILD_RUN_ID,
        "--build-run-id",
        help="Full-dataset build run id to surface.",
    ),
    baseline_eval_run_id: str = typer.Option(
        DEFAULT_BASELINE_EVAL_RUN_ID,
        "--baseline-eval-run-id",
        help="Baseline eval run id to surface.",
    ),
    refresh_modal_apps: bool = typer.Option(
        True,
        "--refresh-modal-apps/--no-refresh-modal-apps",
        help="Run `modal app list --json` while building the dashboard.",
    ),
) -> None:
    """Build a local HTML dashboard for the active training goal."""
    result = write_status_dashboard(
        repo_root=repo_root,
        out_dir=out,
        build_run_id=build_run_id,
        baseline_eval_run_id=baseline_eval_run_id,
        refresh_modal_apps=refresh_modal_apps,
    )
    typer.echo(
        json.dumps(
            {
                "schema_version": result.snapshot["schema_version"],
                "snapshot_path": str(result.snapshot_path),
                "html_path": str(result.html_path),
                "goal_status": result.snapshot["goal_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


@data_app.command("index-altorg")
def index_altorg(
    metadata_root: Path = typer.Option(
        ...,
        "--metadata-root",
        help="Directory containing xlogfile.* and blacklist.txt.",
    ),
    ttyrec_root: Path = typer.Option(
        ...,
        "--ttyrec-root",
        help="Directory containing one player subdirectory per ttyrec owner.",
    ),
    staging_root: Path = typer.Option(
        ...,
        "--staging-root",
        help="Ignored symlink staging directory to present canonical alt.org shape.",
    ),
    db: Path = typer.Option(..., "--db", help="Output NLE ttyrecs.db path."),
    dataset_name: str = typer.Option("nld-nao", "--dataset-name"),
) -> None:
    """Index an alt.org/NAO ttyrec tree into an NLE dataset DB artifact."""
    report = build_altorg_index(
        metadata_root=metadata_root,
        ttyrec_root=ttyrec_root,
        staging_root=staging_root,
        db_path=db,
        dataset_name=dataset_name,
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@data_app.command("write-build-contract")
def write_build_contract(
    db: Path = typer.Option(..., "--db", help="Path to local NLD ttyrecs.db."),
    mode: str = typer.Option("single_frame", "--mode"),
    out: Path = typer.Option(..., "--out", help="Output artifact directory."),
    max_rows: int = typer.Option(1000, "--max-rows"),
    tasks: str = typer.Option("policy_action,next_frame", "--tasks"),
) -> None:
    """Write a build-contract artifact without decoding ttyrecs."""
    report = inspect_nld_db(db)
    task_names = tuple(part.strip() for part in tasks.split(",") if part.strip())
    out.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "learn-nethack.sft-build-cli-contract.v1",
        "db_path": str(db),
        "dataset_name": report.dataset_name,
        "mode": mode,
        "max_rows": max_rows,
        "tasks": list(task_names),
        "status": "requires_nle_dataset_and_action_manifest",
    }
    (out / "build_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    typer.echo(json.dumps(contract, indent=2, sort_keys=True))


@sft_app.command("compare")
def compare_sft_metrics(
    baseline: Path = typer.Option(..., "--baseline", help="Baseline metrics JSON."),
    trained: Path = typer.Option(..., "--trained", help="Trained metrics JSON."),
    out: Path = typer.Option(..., "--out", help="Output score-to-beat report path."),
    baseline_run_id: str | None = typer.Option(None, "--baseline-run-id"),
    trained_run_id: str = typer.Option(..., "--trained-run-id"),
) -> None:
    """Compare baseline and trained metrics and write a score-to-beat report."""
    baseline_metrics = json.loads(baseline.read_text(encoding="utf-8"))
    trained_metrics = json.loads(trained.read_text(encoding="utf-8"))
    report = build_score_to_beat_report(
        baseline_metrics=baseline_metrics,
        trained_metrics=trained_metrics,
        baseline_run_id=baseline_run_id,
        trained_run_id=trained_run_id,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@sft_app.command("proof-gate")
def proof_gate(
    score_to_beat: Path = typer.Option(
        ...,
        "--score-to-beat",
        help="Offline score_to_beat.json from sft compare.",
    ),
    watch_report: Path = typer.Option(
        ...,
        "--watch-report",
        help="Live watch report.json from watch compare.",
    ),
    out: Path = typer.Option(..., "--out", help="Output proof-gate report path."),
) -> None:
    """Combine offline dynamics/policy metrics with live watch fitness."""
    score_report = json.loads(score_to_beat.read_text(encoding="utf-8"))
    watch_payload = json.loads(watch_report.read_text(encoding="utf-8"))
    report = build_training_proof_gate_report(
        score_to_beat_report=score_report,
        watch_report=watch_payload,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@sft_app.command("eval-status")
def eval_status(
    eval_run_id: str = typer.Option(
        ...,
        "--eval-run-id",
        help="Modal run id that writes /runs/<id>/reports/sft_eval_*.json.",
    ),
    local_artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--local-artifact-root",
        help="Local directory where small eval reports should be pulled.",
    ),
    check_remote: bool = typer.Option(
        False,
        "--check-remote",
        help="Pull small report/progress files from the Modal runs volume first.",
    ),
    out: Path | None = typer.Option(None, "--out", help="Optional status report path."),
) -> None:
    """Report whether an SFT eval has completed metrics and report artifacts."""
    remote_status = (
        check_modal_eval_markers(
            eval_run_id=eval_run_id,
            local_artifact_root=local_artifact_root,
        )
        if check_remote
        else None
    )
    report = build_eval_status_report(
        eval_run_id=eval_run_id,
        local_artifact_root=local_artifact_root,
        remote_marker_status=remote_status,
    )
    if out is not None:
        write_eval_status_report(report, out)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@sft_app.command("full-build-followup")
def full_build_followup(
    build_run_id: str = typer.Option(
        ...,
        "--build-run-id",
        help="Modal run id that built /runs/<id>/sft-data.",
    ),
    train_run_id: str | None = typer.Option(
        None,
        "--train-run-id",
        help="Run id for the follow-up sft_train_existing job.",
    ),
    app_id: str | None = typer.Option(
        None,
        "--app-id",
        help="Detached Modal app id to include in log commands.",
    ),
    local_artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--local-artifact-root",
        help="Local directory where small build reports should be pulled.",
    ),
    model_name: str = typer.Option(SftTrainConfig().model_name, "--model-name"),
    max_steps: int = typer.Option(
        0,
        "--max-steps",
        help=(
            "Optimizer steps for sft_train_existing. Use 0 to compute one full "
            "pass over train.jsonl after the completed manifest is available."
        ),
    ),
    training_objective: str = typer.Option(
        SftTrainConfig().training_objective,
        "--training-objective",
        help="SFT objective: policy_dynamics_phased, policy_only, or dynamics_only.",
    ),
    baseline_eval_run_id: str | None = typer.Option(
        None,
        "--baseline-eval-run-id",
        help="Optional run id for the baseline next-n eval command.",
    ),
    trained_eval_run_id: str | None = typer.Option(
        None,
        "--trained-eval-run-id",
        help="Optional run id for the trained next-n eval command.",
    ),
    eval_mode: str = typer.Option(
        "single_frame",
        "--eval-mode",
        help="SFT context mode for matched baseline/trained eval commands.",
    ),
    next_frame_generate_max_rows: int = typer.Option(
        64,
        "--next-frame-generate-max-rows",
        help="Rows for generated single next-frame eval.",
    ),
    next_frame_max_new_tokens: int = typer.Option(
        DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
        "--next-frame-max-new-tokens",
        help="Maximum generated tokens for next-frame eval responses.",
    ),
    next_frame_sequence_max_windows: int = typer.Option(
        64,
        "--next-frame-sequence-max-windows",
        help="Windows per horizon for next-1/5/10 generated sequence eval.",
    ),
    watch_run_id: str | None = typer.Option(
        None,
        "--watch-run-id",
        help="Optional run id for the score/damage watch comparison.",
    ),
    watch_action_manifest: str = typer.Option(
        "/datasets/action_manifest_nethack_v0.json",
        "--watch-action-manifest",
        help="Remote action manifest for deterministic live watch proof.",
    ),
    watch_env_id: str = typer.Option(
        "NetHack-v0",
        "--watch-env-id",
        help="NLE env id for deterministic live watch proof.",
    ),
    watch_character: str = typer.Option(
        "mon-hum-neu-mal",
        "--watch-character",
        help="Fixed character for deterministic live watch proof.",
    ),
    watch_seeds: str = typer.Option(
        DEFAULT_WATCH_PROOF_SEEDS,
        "--watch-seeds",
        help="Comma-separated seeds for live watch proof.",
    ),
    watch_max_steps: int = typer.Option(
        80,
        "--watch-max-steps",
        help="Maximum NLE steps per seed for live watch proof.",
    ),
) -> None:
    """Print commands and local status for a completed full SFT build."""
    effective_train_run_id = train_run_id or f"{build_run_id}-train-existing"
    commands = sft_existing_dataset_followup_commands(
        build_run_id=build_run_id,
        train_run_id=effective_train_run_id,
        app_id=app_id,
        local_artifact_root=str(local_artifact_root),
        model_name=model_name,
        max_steps=max_steps,
        training_objective=training_objective,
        baseline_eval_run_id=baseline_eval_run_id,
        trained_eval_run_id=trained_eval_run_id,
        eval_mode=eval_mode,
        next_frame_max_new_tokens=next_frame_max_new_tokens,
        next_frame_generate_max_rows=next_frame_generate_max_rows,
        next_frame_sequence_max_windows=next_frame_sequence_max_windows,
        watch_run_id=watch_run_id,
        watch_action_manifest=watch_action_manifest,
        watch_env_id=watch_env_id,
        watch_character=watch_character,
        watch_seeds=watch_seeds,
        watch_max_steps=watch_max_steps,
    )
    local_dir = local_artifact_root / build_run_id
    expected_files = {
        "sft_build_report": local_dir / "sft_build_report.json",
        "manifest": local_dir / "manifest.json",
        "rejection_report": local_dir / "rejection_report.json",
        "sample_rows": local_dir / "sample_rows.jsonl",
    }
    status_report_path = local_dir / "full_build_status.json"
    readiness_source = "local_markers"
    if status_report_path.exists():
        readiness = json.loads(status_report_path.read_text(encoding="utf-8"))
        if readiness.get("build_run_id") != build_run_id:
            readiness = build_full_build_readiness_report(
                build_run_id=build_run_id,
                local_artifact_root=local_artifact_root,
            )
        else:
            readiness_source = "status_report"
    else:
        readiness = build_full_build_readiness_report(
            build_run_id=build_run_id,
            local_artifact_root=local_artifact_root,
        )
    train_ready = bool(readiness["train_ready"])
    report = {
        "schema_version": "learn-nethack.sft-full-build-followup.v1",
        "build_run_id": build_run_id,
        "train_run_id": effective_train_run_id,
        "remote_dataset_dir": f"/runs/{build_run_id}/sft-data",
        "local_dir": str(local_dir),
        "local_status": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in expected_files.items()
        },
        "training_gate": {
            "status": "open" if train_ready else "closed",
            "train_ready": train_ready,
            "missing_markers": list(readiness["missing_markers"]),
            "next_action": readiness["next_action"],
            "readiness_source": readiness_source,
            "readiness_report_path": str(status_report_path),
            "readiness_report": readiness,
        },
        "commands": commands,
    }
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@sft_app.command("full-build-status")
def full_build_status(
    build_run_id: str = typer.Option(
        ...,
        "--build-run-id",
        help="Modal run id expected to contain /runs/<id>/sft-data.",
    ),
    local_artifact_root: Path = typer.Option(
        Path("artifacts"),
        "--local-artifact-root",
        help="Local directory where small build markers are or should be stored.",
    ),
    check_remote: bool = typer.Option(
        False,
        "--check-remote",
        help="Check Modal volume markers and pull small marker reports locally.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Optional JSON report output path.",
    ),
) -> None:
    """Report whether a full-dataset SFT build is ready for training."""
    remote_marker_status = (
        check_modal_full_build_markers(
            build_run_id=build_run_id,
            local_artifact_root=local_artifact_root,
        )
        if check_remote
        else None
    )
    report = build_full_build_readiness_report(
        build_run_id=build_run_id,
        local_artifact_root=local_artifact_root,
        remote_marker_status=remote_marker_status,
    )
    if out is not None:
        write_full_build_readiness_report(report, out)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@watch_app.command("compare")
def compare_watch(
    current_checkpoint: str | None = typer.Option(
        None,
        "--current-checkpoint",
        help="Optional LoRA adapter checkpoint for the current model.",
    ),
    action_manifest: Path = typer.Option(
        ...,
        "--action-manifest",
        help="Path to action_manifest.json for the active NLE action IDs.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for events.jsonl, report.json, and index.html.",
    ),
    run_id: str = typer.Option("compare-watch-smoke", "--run-id"),
    model_name: str = typer.Option("google/gemma-4-E4b-it", "--model-name"),
    env_id: str = typer.Option("NetHackChallenge-v0", "--env-id"),
    character: str = typer.Option(DEFAULT_NLE_CHARACTER, "--character"),
    seed: int = typer.Option(20260615, "--seed"),
    max_steps: int = typer.Option(80, "--max-steps"),
    device: str | None = typer.Option(None, "--device"),
    dry_run_contract: bool = typer.Option(
        False,
        "--dry-run-contract",
        help="Write the harness contract without importing NLE or model dependencies.",
    ),
) -> None:
    """Watch a current checkpoint and baseline Gemma act side by side."""
    if dry_run_contract:
        report = write_compare_watch_contract(
            run_id=run_id,
            current_checkpoint=current_checkpoint,
            action_manifest_path=action_manifest,
            out_dir=out,
            model_name=model_name,
            env_id=env_id,
            character=character,
            seed=seed,
            max_steps=max_steps,
        )
    else:
        report = run_checkpoint_compare(
            run_id=run_id,
            current_checkpoint=current_checkpoint,
            action_manifest_path=action_manifest,
            out_dir=out,
            model_name=model_name,
            env_id=env_id,
            character=character,
            seed=seed,
            max_steps=max_steps,
            device=device,
        )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@watch_app.command("sweep")
def compare_watch_sweep(
    current_checkpoint: str | None = typer.Option(
        None,
        "--current-checkpoint",
        help="Optional LoRA adapter checkpoint for the current model.",
    ),
    action_manifest: Path = typer.Option(
        ...,
        "--action-manifest",
        help="Path to action_manifest.json for the active NLE action IDs.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for sweep_report.json and seed subdirectories.",
    ),
    run_id: str = typer.Option("compare-watch-sweep", "--run-id"),
    seeds: str = typer.Option("20260615,20260616,20260617", "--seeds"),
    model_name: str = typer.Option("google/gemma-4-E4b-it", "--model-name"),
    env_id: str = typer.Option("NetHackChallenge-v0", "--env-id"),
    character: str = typer.Option(DEFAULT_NLE_CHARACTER, "--character"),
    max_steps: int = typer.Option(80, "--max-steps"),
    device: str | None = typer.Option(None, "--device"),
    dry_run_contract: bool = typer.Option(
        False,
        "--dry-run-contract",
        help="Write the sweep contract without importing NLE or model dependencies.",
    ),
) -> None:
    """Watch a current checkpoint and baseline Gemma across multiple seeds."""
    if dry_run_contract:
        report = write_compare_watch_sweep_contract(
            run_id=run_id,
            current_checkpoint=current_checkpoint,
            action_manifest_path=action_manifest,
            out_dir=out,
            seeds=seeds,
            model_name=model_name,
            env_id=env_id,
            character=character,
            max_steps=max_steps,
        )
    else:
        report = run_checkpoint_compare_sweep(
            run_id=run_id,
            current_checkpoint=current_checkpoint,
            action_manifest_path=action_manifest,
            out_dir=out,
            seeds=seeds,
            model_name=model_name,
            env_id=env_id,
            character=character,
            max_steps=max_steps,
            device=device,
        )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@watch_app.command("build-preferences")
def build_watch_preferences(
    watch_report: Path = typer.Option(
        ...,
        "--watch-report",
        help="watch compare report.json containing events.",
    ),
    out: Path = typer.Option(..., "--out", help="Output preference JSONL path."),
    report: Path = typer.Option(..., "--report", help="Output build report path."),
) -> None:
    """Build same-prompt action preference rows from watch compare events."""
    payload = json.loads(watch_report.read_text(encoding="utf-8"))
    build_report = write_rollout_preference_jsonl(
        watch_report=payload,
        out_path=out,
        report_path=report,
    )
    typer.echo(json.dumps(build_report, indent=2, sort_keys=True))


@play_app.command("dynamics")
def play_dynamics(
    action_manifest: Path = typer.Option(
        ...,
        "--action-manifest",
        help="Path to action_manifest.json for valid NLE action IDs.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output directory for events.jsonl, report.json, and index.html.",
    ),
    adapter_checkpoint: str | None = typer.Option(
        None,
        "--adapter-checkpoint",
        help="LoRA adapter checkpoint trained on next_frame rows.",
    ),
    run_id: str = typer.Option("dynamics-play", "--run-id"),
    model_name: str = typer.Option("google/gemma-4-E4b-it", "--model-name"),
    initial_frame: Path | None = typer.Option(
        None,
        "--initial-frame",
        help="Text file containing the starting rendered observation frame.",
    ),
    initial_row: Path | None = typer.Option(
        None,
        "--initial-row",
        help="JSONL file containing a next_frame row to use as the start frame.",
    ),
    ground_truth_rows: Path | None = typer.Option(
        None,
        "--ground-truth-rows",
        help=(
            "JSONL next_frame rows whose assistant labels are ground truth for "
            "scripted actions. Defaults to --initial-row when available."
        ),
    ),
    actions: str | None = typer.Option(
        None,
        "--actions",
        help="Comma-separated action IDs for non-interactive scripted play.",
    ),
    max_steps: int = typer.Option(80, "--max-steps"),
    max_new_tokens: int = typer.Option(2048, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    device: str | None = typer.Option(None, "--device"),
    dry_run_contract: bool = typer.Option(
        False,
        "--dry-run-contract",
        help="Write the play contract without importing model dependencies.",
    ),
) -> None:
    """Play the supervised next-frame model as a learned dynamics environment."""
    if dry_run_contract:
        report = write_dynamics_play_contract(
            run_id=run_id,
            adapter_checkpoint=adapter_checkpoint,
            action_manifest_path=action_manifest,
            out_dir=out,
            model_name=model_name,
            initial_frame_path=initial_frame,
            initial_row_path=initial_row,
            ground_truth_rows_path=ground_truth_rows,
            max_steps=max_steps,
        )
    else:
        manifest = load_action_manifest(action_manifest)
        model_spec = DynamicsModelSpec(
            model_name=model_name,
            adapter_checkpoint=adapter_checkpoint,
        )
        predictor = TransformerNextFramePredictor(
            model_spec,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        frame = read_initial_frame(
            initial_frame_path=initial_frame,
            initial_row_path=initial_row,
        )
        if actions is None:
            report = run_interactive_dynamics_session(
                run_id=run_id,
                model_spec=model_spec,
                predictor=predictor,
                initial_frame=frame,
                action_manifest=manifest,
                out_dir=out,
                max_steps=max_steps,
            )
        else:
            try:
                action_ids = parse_action_id_list(actions)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            ground_truth_source = ground_truth_rows or initial_row
            ground_truth_frames = (
                read_ground_truth_frames_from_next_frame_rows(
                    ground_truth_source,
                    max_frames=len(action_ids),
                )
                if ground_truth_source is not None
                else None
            )
            report = run_scripted_dynamics_session(
                run_id=run_id,
                model_spec=model_spec,
                predictor=predictor,
                initial_frame=frame,
                action_ids=action_ids,
                ground_truth_frames=ground_truth_frames,
                action_manifest=manifest,
                out_dir=out,
            )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))
