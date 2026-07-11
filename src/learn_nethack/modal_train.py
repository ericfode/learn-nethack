"""Modal entrypoints for Gemma/NetHack readiness and future training jobs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
import socket
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from learn_nethack.modal_config import (
    APT_PACKAGES,
    DEFAULT_GPU,
    DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
    DEFAULT_WATCH_ENV_ID,
    DEFAULT_WATCH_MODEL_NAME,
    HF_CACHE_MOUNT_PATH,
    MODAL_APP_NAME,
    MODAL_SECRET_NAMES,
    MODAL_SOURCE_MODULES,
    MODAL_TRAIN_PIP_PACKAGES,
    MODAL_VOLUMES,
    PYTHON_VERSION,
    SECRET_ENV_VARS,
    modal_cloud_execution_context,
    modal_function_env,
    modal_hf_cache_env,
    modal_secret_names_for_env,
    modal_volume_mounts,
    normalize_hf_token_env,
    resolve_wandb_mode,
    run_artifact_layout,
)
from learn_nethack.action_manifest import load_action_manifest
from learn_nethack.nld_archive import (
    iter_archive_dataset_batches,
    plan_archive_dataset,
)
from learn_nethack.nld_decode import iter_nld_ttyrec_batches
from learn_nethack.nld_metadata import (
    copy_db_with_rewritten_root,
    inspect_nld_db,
    read_game_metadata,
    read_gameids,
    split_gameids,
)
from learn_nethack.nld_decode import normalize_decoded_batch, normalize_frame_only_batch
from learn_nethack.sft_build import (
    SftBuildProgress,
    build_pseudo_label_sft_from_frame_batches,
    build_sft_from_decoded_batches,
    merge_sft_dataset_shards,
    write_pseudo_label_policy_dataset,
    write_sft_dataset,
)
from learn_nethack.sft_eval import (
    build_score_to_beat_report,
    evaluate_next_frame_rows_with_predictor,
    evaluate_next_frame_rows_with_scorer,
    evaluate_next_frame_sequences_with_predictor,
    evaluate_policy_rows_with_policy,
    summarize_next_frame_sequence_rows,
)
from learn_nethack.sft_integrity import (
    SFT_INTEGRITY_SCHEMA_VERSION,
    audit_sft_dataset,
)
from learn_nethack.sft_train import (
    SftTrainConfig,
    build_sft_jsonl_curriculum_plan,
    build_sft_jsonl_training_plan,
    configure_training_runtime,
    create_unsloth_sft_trainer_from_jsonl,
    get_trainer_assistant_mask_report,
    load_jsonl_rows,
    resolve_jsonl_training_config,
    summarize_trainer_loss_history,
)
from learn_nethack.modal_upload import safe_extract_tar_shard
from learn_nethack.wandb_logging import log_sft_build_to_wandb
from learn_nethack.compare_watch import (
    DEFAULT_NLE_CHARACTER,
    ModelWatchSpec,
    TransformerCandidatePolicy,
    parse_seed_list,
    run_checkpoint_compare,
    run_checkpoint_compare_sweep,
)
from learn_nethack.dynamics_play import DynamicsModelSpec, TransformerNextFramePredictor


BatchIterator = Callable[..., Iterable[dict[str, Any]]]


@dataclass(frozen=True)
class DecodedBatchSource:
    dataset_name: str
    gameids: list[int]
    game_metadata_by_id: dict[int, dict[str, Any]]
    effective_db: str | None
    db_copy_report: dict[str, Any] | None
    archive_manifest: str | None
    archive_shard_count: int | None
    archive_shard_index: int | None
    iter_batches: Callable[[list[int] | None], Iterable[dict[str, Any]]]


try:
    import modal
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised in environments without Modal.
    modal = None  # type: ignore[assignment]


def _build_modal_image() -> Any:
    if modal is None:
        return None

    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install(*APT_PACKAGES)
        .pip_install(*MODAL_TRAIN_PIP_PACKAGES)
        .env(
            {
                **modal_hf_cache_env(),
                "WANDB_DIR": "/runs/wandb",
            }
        )
        .add_local_python_source(*MODAL_SOURCE_MODULES)
    )


def _build_modal_resources(
    env: Mapping[str, str | None] | None = None,
) -> tuple[Any, dict[str, Any], list[Any], dict[str, str]]:
    if modal is None:
        return None, {}, [], {}

    effective_env = os.environ if env is None else env
    app = modal.App(MODAL_APP_NAME)
    volumes = {
        volume.mount_path: modal.Volume.from_name(
            volume.name,
            create_if_missing=True,
        )
        for volume in MODAL_VOLUMES
    }
    secrets = [
        modal.Secret.from_name(secret_name)
        for secret_name in modal_secret_names_for_env(effective_env)
    ]
    function_env = modal_function_env(effective_env)
    return app, volumes, secrets, function_env


def _make_modal_function_callable(function: Any) -> Any:
    if callable(function) or not hasattr(function, "local"):
        return function

    def _call_local(self: Any, *args: Any, **kwargs: Any) -> Any:
        return self.local(*args, **kwargs)

    setattr(type(function), "__call__", _call_local)
    return function


def _commit_mounted_volume(mount_path: str) -> bool:
    should_commit = bool(modal is not None and os.environ.get("MODAL_TASK_ID"))
    if not should_commit:
        return False
    volume = volumes.get(mount_path)
    if volume is None:
        raise RuntimeError(f"Modal volume is not mounted at {mount_path}")
    volume.commit()
    return True


image = _build_modal_image()
app, volumes, secrets, function_env = _build_modal_resources()


def local_readiness_report(
    run_id: str, env: dict[str, str | None] | None = None
) -> dict:
    """Build the readiness report without requiring Modal network access."""
    effective_env = os.environ if env is None else env
    wandb_mode = resolve_wandb_mode(effective_env)
    layout = run_artifact_layout(run_id)

    return {
        "run_id": run_id,
        "modal_app": MODAL_APP_NAME,
        "default_gpu": DEFAULT_GPU,
        "volume_mounts": modal_volume_mounts(),
        "secret_names": list(MODAL_SECRET_NAMES),
        "secret_env_vars": list(SECRET_ENV_VARS),
        "wandb": {
            "mode": wandb_mode.mode,
            "requires_api_key": wandb_mode.requires_api_key,
        },
        "artifact_layout": layout,
        "execution": {
            "backend": "local_static",
            "function_call_id": None,
            "input_id": None,
            "task_id": None,
            "hostname": socket.gethostname(),
        },
        "contracts": {
            "wandb_required": True,
            "nle_is_authority": True,
            "balrog_is_not_trainer": True,
            "model_output_json": '{"action_id": <int>}',
        },
    }


def _write_json(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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


def _assistant_mask_wandb_metrics(
    report: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    scalar_keys = (
        "row_count",
        "input_token_count",
        "prompt_token_count",
        "masked_prompt_token_count",
        "assistant_token_count",
        "supervised_assistant_token_count",
        "assistant_tokens_truncated",
        "truncated_token_count",
        "rows_with_truncated_assistant",
        "supervised_token_fraction",
    )
    for key in scalar_keys:
        value = report.get(key)
        if isinstance(value, int | float):
            metrics[f"{prefix}/{key}"] = float(value)
    per_task = report.get("per_task")
    if isinstance(per_task, Mapping):
        for task, task_report in per_task.items():
            if not isinstance(task_report, Mapping):
                continue
            for key in scalar_keys:
                value = task_report.get(key)
                if isinstance(value, int | float):
                    metrics[f"{prefix}/{task}/{key}"] = float(value)
    return metrics


def _write_report(report: dict) -> Path:
    return _write_json(report["artifact_layout"]["report"], report)


def _decoded_batch_source(
    *,
    db: str | None,
    nle_root: str | None,
    archive_manifest: str | None,
    batch_size: int,
    seq_length: int,
    archive_shard_index: int | None = None,
    staged_db_path: str | Path | None = None,
    batch_iterator: BatchIterator = iter_nld_ttyrec_batches,
) -> DecodedBatchSource:
    db = _none_if_empty(db)
    nle_root = _none_if_empty(nle_root)
    archive_manifest = _none_if_empty(archive_manifest)
    dataset_source = _resolve_dataset_source(
        db=db,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
    )
    if dataset_source == "archive_manifest":
        assert archive_manifest is not None
        archive_plan = plan_archive_dataset(archive_manifest)
        if archive_shard_index is not None and (
            archive_shard_index < 0 or archive_shard_index >= len(archive_plan.shards)
        ):
            raise IndexError(f"archive_shard_index out of range: {archive_shard_index}")

        def _iter_archive_batches(
            selected_gameids: list[int] | None = None,
        ) -> Iterable[dict[str, Any]]:
            return iter_archive_dataset_batches(
                archive_plan,
                selected_gameids=selected_gameids,
                shard_indices=(
                    None if archive_shard_index is None else [archive_shard_index]
                ),
                batch_size=batch_size,
                seq_length=seq_length,
                shuffle=False,
                loop_forever=False,
                batch_iterator=batch_iterator,
            )

        return DecodedBatchSource(
            dataset_name=archive_plan.dataset_name,
            gameids=list(archive_plan.gameids),
            game_metadata_by_id=archive_plan.game_metadata_by_id,
            effective_db=None,
            db_copy_report=None,
            archive_manifest=archive_manifest,
            archive_shard_count=len(archive_plan.shards),
            archive_shard_index=archive_shard_index,
            iter_batches=_iter_archive_batches,
        )

    assert db is not None
    effective_db = db
    db_copy_report = None
    if nle_root is not None:
        if staged_db_path is None:
            raise ValueError("staged_db_path is required when nle_root is set")
        effective_db = str(staged_db_path)
        db_copy_report = copy_db_with_rewritten_root(
            source_db=db,
            target_db=effective_db,
            new_root=nle_root,
        )

    db_report = inspect_nld_db(effective_db)
    gameids = read_gameids(effective_db)

    def _iter_db_batches(
        selected_gameids: list[int] | None = None,
    ) -> Iterable[dict[str, Any]]:
        return batch_iterator(
            dataset_name=db_report.dataset_name,
            batch_size=batch_size,
            seq_length=seq_length,
            dbfilename=effective_db,
            gameids=gameids if selected_gameids is None else selected_gameids,
            shuffle=False,
            loop_forever=False,
        )

    return DecodedBatchSource(
        dataset_name=db_report.dataset_name,
        gameids=gameids,
        game_metadata_by_id=read_game_metadata(effective_db),
        effective_db=effective_db,
        db_copy_report=db_copy_report,
        archive_manifest=None,
        archive_shard_count=None,
        archive_shard_index=None,
        iter_batches=_iter_db_batches,
    )


def local_sft_train_contract(
    *,
    run_id: str,
    db: str | None,
    action_manifest: str,
    nle_root: str | None = None,
    archive_manifest: str | None = None,
    mode: str = "single_frame",
    full_dataset: bool = False,
    max_rows: int = 1000,
    model_name: str = SftTrainConfig().model_name,
    max_steps: int = SftTrainConfig().max_steps,
    batch_size: int = 4,
    seq_length: int = 128,
    seed: int = 20260615,
    tasks: str = "policy_action,next_frame",
    label_source: str = "true_keypress",
) -> dict[str, Any]:
    """Return the durable Modal SFT contract before spending GPU time."""
    db = _none_if_empty(db)
    nle_root = _none_if_empty(nle_root)
    archive_manifest = _none_if_empty(archive_manifest)
    if full_dataset and max_rows != 1000:
        raise ValueError("full_dataset cannot be combined with a custom max_rows")
    dataset_source = _resolve_dataset_source(
        db=db,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
    )

    layout = run_artifact_layout(run_id)
    task_names = [task.strip() for task in tasks.split(",") if task.strip()]
    normalized_label_source = _parse_training_label_source(
        label_source=label_source,
        task_names=task_names,
    )
    return {
        "schema_version": "learn-nethack.sft-train-contract.v1",
        "run_id": run_id,
        "modal_app": MODAL_APP_NAME,
        "dataset": {
            "source": dataset_source,
            "db": db,
            "nle_root": nle_root,
            "archive_manifest": archive_manifest,
            "action_manifest": action_manifest,
            "mode": mode,
            "full_dataset": full_dataset,
            "max_rows": None if full_dataset else max_rows,
            "tasks": task_names,
            "label_source": normalized_label_source,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "seed": seed,
        },
        "training": {
            "model_name": model_name,
            "max_steps": max_steps,
            "max_steps_mode": ("auto_one_full_pass" if max_steps <= 0 else "explicit"),
            "trainer": "unsloth_trl_sft",
            "loss_mask": "explicit_final_assistant_tokens",
            "output_contract": '{"action_id": <int>}',
            "auxiliary_task": "raw rendered next-frame text",
        },
        "artifacts": {
            "root": layout["root"],
            "sft_data": str(Path(layout["root"]) / "sft-data"),
            "adapter": layout["adapter"],
            "report": str(Path(layout["root"]) / "reports" / "sft_train_report.json"),
            "contract": str(
                Path(layout["root"]) / "reports" / "sft_train_contract.json"
            ),
            "wandb": layout["wandb"],
        },
        "proof_steps": _sft_proof_steps(dataset_source, normalized_label_source),
    }


def local_sft_train_existing_contract(
    *,
    run_id: str,
    dataset_dir: str,
    model_name: str = SftTrainConfig().model_name,
    max_steps: int = SftTrainConfig().max_steps,
    training_objective: str = SftTrainConfig().training_objective,
) -> dict[str, Any]:
    """Return the Modal SFT contract for an already-built JSONL dataset."""
    dataset_dir = _none_if_empty(dataset_dir)
    if dataset_dir is None:
        raise ValueError("dataset_dir is required")
    dataset_root = Path(dataset_dir)
    layout = run_artifact_layout(run_id)
    return {
        "schema_version": "learn-nethack.sft-train-existing-contract.v1",
        "run_id": run_id,
        "modal_app": MODAL_APP_NAME,
        "dataset": {
            "source": "existing_sft_jsonl",
            "dataset_dir": str(dataset_root),
            "train_file": str(dataset_root / "train.jsonl"),
            "manifest": str(dataset_root / "manifest.json"),
            "rejection_report": str(dataset_root / "rejection_report.json"),
        },
        "training": {
            "model_name": model_name,
            "max_steps": max_steps,
            "max_steps_mode": ("auto_one_full_pass" if max_steps <= 0 else "explicit"),
            "training_objective": training_objective,
            "trainer": "unsloth_trl_sft",
            "loss_mask": "explicit_final_assistant_tokens",
            "output_contract": '{"action_id": <int>}',
            "auxiliary_task": "raw rendered next-frame text",
        },
        "artifacts": {
            "root": layout["root"],
            "adapter": layout["adapter"],
            "report": str(
                Path(layout["root"]) / "reports" / "sft_train_existing_report.json"
            ),
            "contract": str(
                Path(layout["root"]) / "reports" / "sft_train_existing_contract.json"
            ),
            "wandb": layout["wandb"],
        },
        "proof_steps": [
            "verify_existing_sft_rows",
            "train_adapter",
            "eval_baseline_policy_and_next_frame",
            "eval_trained_policy_and_next_frame",
            "compare_policy_and_next_frame",
        ],
    }


def local_sft_build_contract(
    *,
    run_id: str,
    db: str | None,
    action_manifest: str,
    nle_root: str | None = None,
    archive_manifest: str | None = None,
    archive_shard_index: int | None = None,
    mode: str = "single_frame",
    full_dataset: bool = False,
    max_rows: int = 1000,
    batch_size: int = 4,
    seq_length: int = 128,
    seed: int = 20260615,
    tasks: str = "policy_action,next_frame",
    label_source: str = "true_keypress",
) -> dict[str, Any]:
    """Return the durable Modal SFT data-build contract without trainer state."""
    db = _none_if_empty(db)
    nle_root = _none_if_empty(nle_root)
    archive_manifest = _none_if_empty(archive_manifest)
    if full_dataset and max_rows != 1000:
        raise ValueError("full_dataset cannot be combined with a custom max_rows")
    dataset_source = _resolve_dataset_source(
        db=db,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
    )
    if archive_shard_index is not None and dataset_source != "archive_manifest":
        raise ValueError("archive_shard_index requires archive_manifest")
    task_names = [task.strip() for task in tasks.split(",") if task.strip()]
    normalized_label_source = _parse_training_label_source(
        label_source=label_source,
        task_names=task_names,
    )
    layout = run_artifact_layout(run_id)
    return {
        "schema_version": "learn-nethack.sft-build-contract.v1",
        "run_id": run_id,
        "modal_app": MODAL_APP_NAME,
        "dataset": {
            "source": dataset_source,
            "db": db,
            "nle_root": nle_root,
            "archive_manifest": archive_manifest,
            "archive_shard_index": archive_shard_index,
            "action_manifest": action_manifest,
            "mode": mode,
            "full_dataset": full_dataset,
            "max_rows": None if full_dataset else max_rows,
            "tasks": task_names,
            "label_source": normalized_label_source,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "seed": seed,
        },
        "artifacts": {
            "root": layout["root"],
            "sft_data": str(Path(layout["root"]) / "sft-data"),
            "report": str(Path(layout["root"]) / "reports" / "sft_build_report.json"),
            "contract": str(
                Path(layout["root"]) / "reports" / "sft_build_contract.json"
            ),
            "wandb": layout["wandb"],
        },
        "proof_steps": [
            (
                "build_pseudo_label_sft_rows_from_archive_shard"
                if archive_shard_index is not None
                and dataset_source == "archive_manifest"
                and normalized_label_source == "pseudo_visible_player_delta"
                else _sft_proof_steps(dataset_source, normalized_label_source)[0]
            )
        ],
    }


def local_sft_merge_shards_contract(
    *,
    run_id: str,
    shard_run_ids: str | Sequence[str],
) -> dict[str, Any]:
    """Return the durable contract for merging completed SFT shard builds."""
    parsed_shard_run_ids = _parse_shard_run_ids(shard_run_ids)
    layout = run_artifact_layout(run_id)
    return {
        "schema_version": "learn-nethack.sft-merge-shards-contract.v1",
        "run_id": run_id,
        "modal_app": MODAL_APP_NAME,
        "shard_run_ids": parsed_shard_run_ids,
        "shard_dataset_dirs": [
            f"/runs/{shard_run_id}/sft-data" for shard_run_id in parsed_shard_run_ids
        ],
        "artifacts": {
            "root": layout["root"],
            "sft_data": str(Path(layout["root"]) / "sft-data"),
            "report": str(Path(layout["root"]) / "reports" / "sft_merge_report.json"),
            "contract": str(
                Path(layout["root"]) / "reports" / "sft_merge_contract.json"
            ),
            "wandb": layout["wandb"],
        },
        "proof_steps": [
            "verify_completed_sft_shards",
            "merge_shard_jsonl_files",
            "write_trainable_sft_manifest",
        ],
    }


def local_sft_eval_contract(
    *,
    run_id: str,
    db: str | None,
    action_manifest: str,
    nle_root: str | None = None,
    archive_manifest: str | None = None,
    split: str = "validation",
    model_role: str = "baseline",
    adapter: str | None = None,
    model_name: str = SftTrainConfig().model_name,
    mode: str = "single_frame",
    max_rows: int = 128,
    batch_size: int = 4,
    seq_length: int = 128,
    seed: int = 20260615,
    eval_tasks: str = "policy_action,next_frame",
    label_source: str = "true_keypress",
    next_frame_eval_mode: str = "teacher_forced",
    next_frame_max_new_tokens: int = DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
    next_frame_generate_max_rows: int = 64,
    next_frame_sequence_horizons: str = "1,5,10",
    next_frame_sequence_max_windows: int = 64,
) -> dict[str, Any]:
    """Return the Modal SFT eval contract for baseline or trained scoring."""
    db = _none_if_empty(db)
    nle_root = _none_if_empty(nle_root)
    archive_manifest = _none_if_empty(archive_manifest)
    dataset_source = _resolve_dataset_source(
        db=db,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
    )
    task_names = _parse_task_names(eval_tasks)
    normalized_label_source = _parse_training_label_source(
        label_source=label_source,
        task_names=list(task_names),
    )
    frame_eval_mode = _parse_next_frame_eval_mode(next_frame_eval_mode)
    sequence_horizons = _parse_next_frame_sequence_horizons(
        next_frame_sequence_horizons
    )
    if next_frame_sequence_max_windows <= 0:
        raise ValueError("next_frame_sequence_max_windows must be positive")
    if next_frame_generate_max_rows <= 0:
        raise ValueError("next_frame_generate_max_rows must be positive")
    required_metrics = []
    if "policy_action" in task_names:
        required_metrics.extend(
            ["parse_valid_rate", "action_space_valid_rate", "exact_match_rate"]
        )
    if "next_frame" in task_names:
        required_metrics.extend(
            [
                "next_frame_sequence_row_count",
                "next_frame_sequence_episode_count",
                "next_frame_sequence_segment_count",
                "next_frame_sequence_max_segment_length",
            ]
        )
        for horizon in sequence_horizons:
            required_metrics.extend(
                [
                    f"next_{horizon}_frame_sequence_available_window_count",
                    f"next_{horizon}_frame_sequence_available_frame_count",
                    f"next_{horizon}_frame_sequence_eligible_segment_count",
                ]
            )
        if frame_eval_mode in {"teacher_forced", "both"}:
            required_metrics.extend(
                [
                    "next_frame_teacher_forced_mean_nll",
                    "next_frame_teacher_forced_token_accuracy",
                    "next_frame_teacher_forced_perplexity",
                ]
            )
        if frame_eval_mode in {"generate", "both"}:
            required_metrics.extend(
                [
                    "next_frame_parse_valid_rate",
                    "next_frame_char_accuracy",
                    "next_frame_exact_match_rate",
                ]
            )
            for horizon in sequence_horizons:
                required_metrics.extend(
                    [
                        f"next_{horizon}_frame_sequence_parse_valid_rate",
                        f"next_{horizon}_frame_sequence_window_count",
                        f"next_{horizon}_frame_sequence_frame_count",
                        f"next_{horizon}_frame_sequence_char_accuracy",
                        f"next_{horizon}_frame_sequence_exact_match_rate",
                    ]
                )
    layout = run_artifact_layout(run_id)
    return {
        "schema_version": "learn-nethack.sft-eval-contract.v1",
        "run_id": run_id,
        "model": {
            "role": model_role,
            "model_name": model_name,
            "adapter": adapter,
        },
        "dataset": {
            "source": dataset_source,
            "db": db,
            "nle_root": nle_root,
            "archive_manifest": archive_manifest,
            "action_manifest": action_manifest,
            "split": split,
            "mode": mode,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "seed": seed,
            "label_source": normalized_label_source,
        },
        "evaluation": {
            "tasks": list(task_names),
            "next_frame_eval_mode": frame_eval_mode,
            "next_frame_max_new_tokens": next_frame_max_new_tokens,
            "next_frame_generate_max_rows": next_frame_generate_max_rows,
            "next_frame_sequence_horizons": list(sequence_horizons),
            "next_frame_sequence_max_windows": next_frame_sequence_max_windows,
        },
        "artifacts": {
            "root": layout["root"],
            "metrics": str(Path(layout["root"]) / "reports" / "sft_eval_metrics.json"),
            "report": str(Path(layout["root"]) / "reports" / "sft_eval_report.json"),
            "progress": str(
                Path(layout["root"]) / "reports" / "sft_eval_progress.jsonl"
            ),
            "contract": str(
                Path(layout["root"]) / "reports" / "sft_eval_contract.json"
            ),
            "wandb": layout["wandb"],
            "watch": layout["watch"],
        },
        "required_metrics": required_metrics,
    }


def _parse_task_names(tasks: str) -> tuple[str, ...]:
    task_names = tuple(part.strip() for part in tasks.split(",") if part.strip())
    if not task_names:
        raise ValueError("at least one task is required")
    allowed = {"policy_action", "next_frame"}
    unknown = sorted(set(task_names) - allowed)
    if unknown:
        raise ValueError(f"unknown eval task(s): {unknown}")
    return task_names


def _parse_next_frame_eval_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    allowed = {"teacher_forced", "generate", "both"}
    if normalized not in allowed:
        raise ValueError(f"unknown next-frame eval mode: {mode!r}")
    return normalized


def _parse_next_frame_sequence_horizons(horizons: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in horizons.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            horizon = int(stripped)
        except ValueError as exc:
            raise ValueError(f"invalid next-frame sequence horizon: {part!r}") from exc
        if horizon <= 0:
            raise ValueError(f"next-frame sequence horizon must be positive: {horizon}")
        values.append(horizon)
    if not values:
        raise ValueError("at least one next-frame sequence horizon is required")
    return tuple(values)


def local_watch_compare_contract(
    *,
    run_id: str,
    action_manifest: str,
    current_checkpoint: str | None = None,
    model_name: str = DEFAULT_WATCH_MODEL_NAME,
    env_id: str = DEFAULT_WATCH_ENV_ID,
    character: str = DEFAULT_NLE_CHARACTER,
    seed: int = 20260615,
    max_steps: int = 10,
    device: str | None = None,
) -> dict[str, Any]:
    """Return the Modal watch-compare contract before model execution."""
    layout = run_artifact_layout(run_id)
    watch_dir = layout["watch"]
    return {
        "schema_version": "learn-nethack.watch-compare-contract.v1",
        "run_id": run_id,
        "modal_app": MODAL_APP_NAME,
        "watch": {
            "action_manifest": action_manifest,
            "current_checkpoint": _none_if_empty(current_checkpoint),
            "model_name": model_name,
            "env_id": env_id,
            "character": character,
            "seed": seed,
            "max_steps": max_steps,
            "device": device,
        },
        "artifacts": {
            "root": layout["root"],
            "watch_dir": watch_dir,
            "events": str(Path(watch_dir) / "events.jsonl"),
            "latest": str(Path(watch_dir) / "latest.json"),
            "viewer": str(Path(watch_dir) / "index.html"),
            "report": str(Path(watch_dir) / "report.json"),
            "contract": str(
                Path(layout["root"]) / "reports" / "watch_compare_contract.json"
            ),
        },
    }


def local_watch_compare_sweep_contract(
    *,
    run_id: str,
    action_manifest: str,
    current_checkpoint: str | None = None,
    model_name: str = DEFAULT_WATCH_MODEL_NAME,
    env_id: str = DEFAULT_WATCH_ENV_ID,
    character: str = DEFAULT_NLE_CHARACTER,
    seeds: str | Iterable[int] = "20260615,20260616,20260617",
    max_steps: int = 10,
    device: str | None = None,
) -> dict[str, Any]:
    """Return the Modal watch-sweep contract before model execution."""
    layout = run_artifact_layout(run_id)
    watch_dir = layout["watch"]
    seed_values = parse_seed_list(list(seeds) if not isinstance(seeds, str) else seeds)
    return {
        "schema_version": "learn-nethack.watch-compare-sweep-contract.v1",
        "run_id": run_id,
        "modal_app": MODAL_APP_NAME,
        "watch": {
            "action_manifest": action_manifest,
            "current_checkpoint": _none_if_empty(current_checkpoint),
            "model_name": model_name,
            "env_id": env_id,
            "character": character,
            "seeds": seed_values,
            "max_steps": max_steps,
            "device": device,
        },
        "artifacts": {
            "root": layout["root"],
            "watch_dir": watch_dir,
            "report": str(Path(watch_dir) / "sweep_report.json"),
            "contract": str(
                Path(layout["root"]) / "reports" / "watch_compare_sweep_contract.json"
            ),
            "seed_dirs": [
                str(Path(watch_dir) / f"seed-{seed}") for seed in seed_values
            ],
        },
    }


def _none_if_empty(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _parse_shard_run_ids(shard_run_ids: str | Sequence[str]) -> list[str]:
    if isinstance(shard_run_ids, str):
        values = [part.strip() for part in shard_run_ids.split(",")]
    else:
        values = [str(part).strip() for part in shard_run_ids]
    parsed = [value for value in values if value]
    if not parsed:
        raise ValueError("at least one shard run id is required")
    return parsed


def _resolve_dataset_source(
    *,
    db: str | None,
    nle_root: str | None,
    archive_manifest: str | None,
) -> str:
    if archive_manifest is not None:
        if db is not None or nle_root is not None:
            raise ValueError(
                "archive_manifest dataset source cannot be combined with db or nle_root"
            )
        return "archive_manifest"
    if db is None:
        raise ValueError("SFT runs require a dataset source: db or archive_manifest")
    return "nld_db"


def _parse_training_label_source(*, label_source: str, task_names: list[str]) -> str:
    if not task_names:
        raise ValueError("at least one SFT task is required")
    allowed_tasks = {"policy_action", "next_frame"}
    unknown_tasks = sorted(set(task_names) - allowed_tasks)
    if unknown_tasks:
        raise ValueError(f"unknown SFT task(s): {unknown_tasks}")
    normalized = label_source.strip().lower()
    allowed = {"true_keypress", "pseudo_visible_player_delta"}
    if normalized not in allowed:
        raise ValueError(f"unknown training label_source: {label_source!r}")
    return normalized


def _sft_proof_steps(dataset_source: str, label_source: str) -> list[str]:
    if label_source == "pseudo_visible_player_delta":
        build_step = (
            "build_pseudo_label_sft_rows_from_archive_shards"
            if dataset_source == "archive_manifest"
            else "build_pseudo_label_sft_rows"
        )
    else:
        build_step = (
            "build_full_sft_rows_from_archive_shards"
            if dataset_source == "archive_manifest"
            else "build_full_sft_rows"
        )
    return [
        build_step,
        "train_adapter",
        "eval_baseline_policy_and_next_frame",
        "eval_trained_policy_and_next_frame",
        "compare_policy_and_next_frame",
    ]


def _sft_train_impl(
    *,
    run_id: str,
    db: str | None,
    action_manifest: str,
    nle_root: str | None = None,
    archive_manifest: str | None = None,
    mode: str = "single_frame",
    full_dataset: bool = False,
    max_rows: int = 1000,
    batch_size: int = 4,
    seq_length: int = 128,
    seed: int = 20260615,
    tasks: str = "policy_action,next_frame",
    label_source: str = "true_keypress",
    model_name: str = SftTrainConfig().model_name,
    max_steps: int = SftTrainConfig().max_steps,
    dry_run_contract: bool = False,
) -> dict[str, Any]:
    normalize_hf_token_env(os.environ)
    contract = local_sft_train_contract(
        run_id=run_id,
        db=db,
        action_manifest=action_manifest,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
        mode=mode,
        full_dataset=full_dataset,
        max_rows=max_rows,
        batch_size=batch_size,
        seq_length=seq_length,
        seed=seed,
        tasks=tasks,
        label_source=label_source,
        model_name=model_name,
        max_steps=max_steps,
    )
    contract_path = _write_json(contract["artifacts"]["contract"], contract)
    if dry_run_contract:
        return {
            **contract,
            "status": "contract_written",
            "contract_path": str(contract_path),
        }

    wandb_mode = resolve_wandb_mode(os.environ)
    source = _decoded_batch_source(
        db=db,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
        batch_size=batch_size,
        seq_length=seq_length,
        staged_db_path=Path(contract["artifacts"]["root"]) / "staged_ttyrecs.db",
    )
    task_names = tuple(contract["dataset"]["tasks"])
    effective_max_rows = contract["dataset"]["max_rows"]
    manifest = load_action_manifest(action_manifest)
    sft_data_dir = Path(contract["artifacts"]["sft_data"])
    build_result = _build_sft_rows_for_contract(
        contract=contract,
        source=source,
        action_manifest=manifest,
        out_dir=sft_data_dir,
    )
    train_label_source = str(contract["dataset"]["label_source"])
    build_wandb_mode = log_sft_build_to_wandb(
        output_dir=sft_data_dir,
        metrics={
            "accepted_policy_rows": build_result.accepted_policy_rows,
            "accepted_next_frame_rows": build_result.accepted_next_frame_rows,
            "rejected_rows": build_result.rejected_rows,
        },
        config={
            "run_id": run_id,
            "dataset_name": source.dataset_name,
            "dataset_source": contract["dataset"]["source"],
            "db": source.effective_db,
            "source_db": db,
            "nle_root": nle_root,
            "archive_manifest": source.archive_manifest,
            "archive_shard_count": source.archive_shard_count,
            "mode": mode,
            "tasks": list(task_names),
            "label_source": train_label_source,
            "full_dataset": full_dataset,
            "max_rows": effective_max_rows,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "seed": seed,
        },
        env=os.environ,
        run_name=f"{run_id}-sft-data-build",
    )
    train_config = SftTrainConfig(model_name=model_name, max_steps=max_steps)
    train_plan = build_sft_jsonl_training_plan(
        dataset_dir=sft_data_dir,
        output_dir=contract["artifacts"]["adapter"],
        config=train_config,
    )
    hf_cache_committed_after_model_load = False
    try:
        trainer = create_unsloth_sft_trainer_from_jsonl(
            jsonl_paths=train_plan["train_files"],
            output_dir=contract["artifacts"]["adapter"],
            config=train_config,
            env=os.environ,
        )
    finally:
        hf_cache_committed_after_model_load = _commit_mounted_volume(
            HF_CACHE_MOUNT_PATH
        )
    assistant_mask_report = get_trainer_assistant_mask_report(trainer)
    configure_training_runtime(train_config)
    train_output = trainer.train()
    trainer.model.save_pretrained(contract["artifacts"]["adapter"])
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is not None:
        processing_class.save_pretrained(contract["artifacts"]["adapter"])
    report = {
        "schema_version": "learn-nethack.sft-train-report.v2",
        "run_id": run_id,
        "status": "completed",
        "contract_path": str(contract_path),
        "db_copy_report": source.db_copy_report,
        "archive_manifest": source.archive_manifest,
        "archive_shard_count": source.archive_shard_count,
        "wandb_mode": wandb_mode.mode,
        "build_wandb_mode": build_wandb_mode,
        "build_result": build_result.__dict__,
        "training_plan": train_plan,
        "assistant_mask": assistant_mask_report,
        "train_metrics": getattr(train_output, "metrics", {}),
        "hf_cache": {
            "mount_path": HF_CACHE_MOUNT_PATH,
            "committed_after_model_load": hf_cache_committed_after_model_load,
        },
    }
    report_path = _write_json(contract["artifacts"]["report"], report)
    import wandb

    run = wandb.run or wandb.init(
        project="learn-nethack",
        name=run_id,
        job_type="sft-train",
        mode=wandb_mode.mode,
        config=report,
        dir=contract["artifacts"]["wandb"],
    )
    report["wandb"] = _wandb_run_report(run, wandb_mode.mode)
    report_path = _write_json(contract["artifacts"]["report"], report)
    run.log(
        {
            "sft_train/accepted_policy_rows": build_result.accepted_policy_rows,
            "sft_train/accepted_next_frame_rows": build_result.accepted_next_frame_rows,
            "sft_train/rejected_rows": build_result.rejected_rows,
            **_assistant_mask_wandb_metrics(
                assistant_mask_report,
                prefix="sft_train/assistant_mask",
            ),
        }
    )
    adapter_artifact = wandb.Artifact(name=f"sft-adapter-{run_id}", type="model")
    adapter_artifact.add_dir(contract["artifacts"]["adapter"])
    adapter_artifact.add_file(str(report_path))
    adapter_artifact.add_file(str(contract_path))
    run.log_artifact(adapter_artifact)
    run.finish()
    return {**report, "report_path": str(report_path)}


def _sft_train_existing_impl(
    *,
    run_id: str,
    dataset_dir: str,
    model_name: str = SftTrainConfig().model_name,
    max_steps: int = SftTrainConfig().max_steps,
    training_objective: str = SftTrainConfig().training_objective,
    dry_run_contract: bool = False,
) -> dict[str, Any]:
    normalize_hf_token_env(os.environ)
    contract = local_sft_train_existing_contract(
        run_id=run_id,
        dataset_dir=dataset_dir,
        model_name=model_name,
        max_steps=max_steps,
        training_objective=training_objective,
    )
    contract_path = _write_json(contract["artifacts"]["contract"], contract)
    if dry_run_contract:
        return {
            **contract,
            "status": "contract_written",
            "contract_path": str(contract_path),
        }

    wandb_mode = resolve_wandb_mode(os.environ)
    dataset_summary = _existing_sft_dataset_summary(
        Path(contract["dataset"]["dataset_dir"])
    )
    requested_train_config = SftTrainConfig(
        model_name=model_name,
        max_steps=max_steps,
        training_objective=training_objective,
    )
    resolved_training = resolve_jsonl_training_config(
        dataset_dir=contract["dataset"]["dataset_dir"],
        config=requested_train_config,
    )
    train_config = resolved_training.config
    train_plan = build_sft_jsonl_curriculum_plan(
        dataset_dir=contract["dataset"]["dataset_dir"],
        output_dir=contract["artifacts"]["adapter"],
        scratch_dir=Path(contract["artifacts"]["root"]) / "curriculum-data",
        config=train_config,
        requested_max_steps=resolved_training.requested_max_steps,
    )
    train_plan.update(
        {
            "max_steps_mode": resolved_training.max_steps_mode,
            "requested_max_steps": resolved_training.requested_max_steps,
            "train_row_count": resolved_training.train_row_count,
            "effective_batch_size": resolved_training.effective_batch_size,
        }
    )
    import wandb

    run = wandb.run or wandb.init(
        project="learn-nethack",
        name=run_id,
        job_type="sft-train-existing",
        mode=wandb_mode.mode,
        config={
            "schema_version": "learn-nethack.sft-train-existing-run-config.v1",
            "run_id": run_id,
            "dataset": dataset_summary,
            "training_plan": train_plan,
        },
        dir=contract["artifacts"]["wandb"],
    )
    hf_cache_committed_after_model_load = False
    trainer: Any | None = None
    model: Any | None = None
    tokenizer: Any | None = None
    phase_reports: list[dict[str, Any]] = []
    phases = train_plan.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("SFT training plan has no phases")
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError(f"invalid SFT phase entry: {phase!r}")
        phase_name = str(phase["name"])
        phase_steps = int(phase["max_steps"])
        phase_config = replace(train_config, max_steps=phase_steps)
        trainer = create_unsloth_sft_trainer_from_jsonl(
            jsonl_paths=phase["train_files"],
            output_dir=contract["artifacts"]["adapter"],
            config=phase_config,
            env=os.environ,
            model=model,
            tokenizer=tokenizer,
        )
        if not hf_cache_committed_after_model_load:
            hf_cache_committed_after_model_load = _commit_mounted_volume(
                HF_CACHE_MOUNT_PATH
            )
        configure_training_runtime(phase_config)
        train_output = trainer.train()
        model = trainer.model
        tokenizer = getattr(trainer, "processing_class", tokenizer)
        phase_metrics = getattr(train_output, "metrics", {})
        assistant_mask_report = get_trainer_assistant_mask_report(trainer)
        trainer_state = getattr(trainer, "state", None)
        loss_history = summarize_trainer_loss_history(
            getattr(trainer_state, "log_history", [])
        )
        phase_reports.append(
            {
                "name": phase_name,
                "max_steps": phase_steps,
                "train_files": list(phase["train_files"]),
                "row_count": phase.get("row_count"),
                "tasks": phase.get("tasks"),
                "assistant_mask": assistant_mask_report,
                "loss_history": loss_history,
                "metrics": phase_metrics,
            }
        )
        run.log(
            {
                f"sft_train_existing/{phase_name}/{key}": float(value)
                for key, value in phase_metrics.items()
                if isinstance(value, int | float)
            }
        )
        run.log(
            _assistant_mask_wandb_metrics(
                assistant_mask_report,
                prefix=f"sft_train_existing/{phase_name}/assistant_mask",
            )
        )
        run.log(
            {
                f"sft_train_existing/{phase_name}/overfit/{key}": float(value)
                for key, value in loss_history.items()
                if isinstance(value, int | float | bool)
            }
        )
    if trainer is None:
        raise RuntimeError("SFT training produced no trainer")
    trainer.model.save_pretrained(contract["artifacts"]["adapter"])
    processing_class = getattr(trainer, "processing_class", None)
    if processing_class is not None:
        processing_class.save_pretrained(contract["artifacts"]["adapter"])
    report = {
        "schema_version": "learn-nethack.sft-train-existing-report.v2",
        "run_id": run_id,
        "status": "completed",
        "contract_path": str(contract_path),
        "wandb_mode": wandb_mode.mode,
        "dataset": dataset_summary,
        "training_plan": train_plan,
        "phase_reports": phase_reports,
        "train_metrics": phase_reports[-1]["metrics"] if phase_reports else {},
        "hf_cache": {
            "mount_path": HF_CACHE_MOUNT_PATH,
            "committed_after_model_load": hf_cache_committed_after_model_load,
        },
    }
    report_path = _write_json(contract["artifacts"]["report"], report)
    report["wandb"] = _wandb_run_report(run, wandb_mode.mode)
    report_path = _write_json(contract["artifacts"]["report"], report)
    run.config.update({"final_report": report}, allow_val_change=True)
    run.log(_existing_sft_dataset_wandb_metrics(dataset_summary))
    adapter_artifact = wandb.Artifact(name=f"sft-adapter-{run_id}", type="model")
    adapter_artifact.add_dir(contract["artifacts"]["adapter"])
    adapter_artifact.add_file(str(report_path))
    adapter_artifact.add_file(str(contract_path))
    for optional_path in (
        dataset_summary.get("manifest_path"),
        dataset_summary.get("rejection_report_path"),
        dataset_summary.get("integrity_report_path"),
    ):
        if optional_path and Path(str(optional_path)).exists():
            adapter_artifact.add_file(str(optional_path))
    run.log_artifact(adapter_artifact)
    run.finish()
    return {**report, "report_path": str(report_path)}


def _existing_sft_dataset_summary(dataset_dir: Path) -> dict[str, Any]:
    train_file = dataset_dir / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"required SFT train file is missing: {train_file}")
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"required completed SFT manifest is missing: {manifest_path}"
        )
    rejection_report_path = dataset_dir / "rejection_report.json"
    if not rejection_report_path.exists():
        raise FileNotFoundError(
            "required completed SFT rejection report is missing: "
            f"{rejection_report_path}"
        )
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "train_file": str(train_file),
        "manifest_path": str(manifest_path),
        "rejection_report_path": str(rejection_report_path),
    }
    summary["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = summary["manifest"]
    if (
        isinstance(manifest, Mapping)
        and manifest.get("split_row_limits") is not None
        and manifest.get("split_limits_satisfied") is not True
    ):
        raise RuntimeError(
            f"SFT dataset split row limits are incomplete: {manifest_path}"
        )
    if isinstance(manifest, Mapping) and manifest.get("split_row_limits") is not None:
        integrity_report_path = dataset_dir / "integrity_report.json"
        if not integrity_report_path.exists():
            raise FileNotFoundError(
                "required corrected-dataset integrity report is missing: "
                f"{integrity_report_path}"
            )
        recorded_integrity = json.loads(
            integrity_report_path.read_text(encoding="utf-8")
        )
        if not isinstance(recorded_integrity, Mapping) or (
            recorded_integrity.get("passed") is not True
        ):
            raise RuntimeError(
                "recorded SFT dataset integrity audit did not pass: "
                f"{integrity_report_path}"
            )
        if recorded_integrity.get("schema_version") != SFT_INTEGRITY_SCHEMA_VERSION:
            raise RuntimeError(
                "recorded SFT dataset integrity schema is obsolete: "
                f"{recorded_integrity.get('schema_version')!r}"
            )
        live_integrity = audit_sft_dataset(
            dataset_dir,
            expected_env_id=str(manifest.get("env_id")),
        )
        if live_integrity.get("passed") is not True:
            raise RuntimeError(
                "live SFT dataset integrity audit failed: "
                f"{live_integrity.get('failure_reasons')}"
            )
        if recorded_integrity.get("file_fingerprints") != live_integrity.get(
            "file_fingerprints"
        ):
            raise RuntimeError(
                "SFT dataset files changed after the recorded integrity audit"
            )
        summary["integrity_report_path"] = str(integrity_report_path)
        summary["integrity_report"] = live_integrity
    summary["rejection_report"] = json.loads(
        rejection_report_path.read_text(encoding="utf-8")
    )
    return summary


def _existing_sft_dataset_wandb_metrics(
    dataset_summary: Mapping[str, Any],
) -> dict[str, float]:
    manifest = dataset_summary.get("manifest")
    if not isinstance(manifest, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key in (
        "accepted_policy_rows",
        "accepted_next_frame_rows",
        "rejected_rows",
    ):
        value = manifest.get(key)
        if isinstance(value, int | float):
            metrics[f"sft_train_existing/{key}"] = float(value)
    return metrics


def _sft_build_impl(
    *,
    run_id: str,
    db: str | None,
    action_manifest: str,
    nle_root: str | None = None,
    archive_manifest: str | None = None,
    archive_shard_index: int | None = None,
    mode: str = "single_frame",
    full_dataset: bool = False,
    max_rows: int = 1000,
    batch_size: int = 4,
    seq_length: int = 128,
    seed: int = 20260615,
    tasks: str = "policy_action,next_frame",
    label_source: str = "true_keypress",
    dry_run_contract: bool = False,
) -> dict[str, Any]:
    contract = local_sft_build_contract(
        run_id=run_id,
        db=db,
        action_manifest=action_manifest,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
        archive_shard_index=archive_shard_index,
        mode=mode,
        full_dataset=full_dataset,
        max_rows=max_rows,
        batch_size=batch_size,
        seq_length=seq_length,
        seed=seed,
        tasks=tasks,
        label_source=label_source,
    )
    contract_path = _write_json(contract["artifacts"]["contract"], contract)
    if dry_run_contract:
        return {
            **contract,
            "status": "contract_written",
            "contract_path": str(contract_path),
        }

    wandb_mode = resolve_wandb_mode(os.environ)
    source = _decoded_batch_source(
        db=db,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
        archive_shard_index=archive_shard_index,
        batch_size=batch_size,
        seq_length=seq_length,
        staged_db_path=Path(contract["artifacts"]["root"]) / "staged_ttyrecs.db",
    )
    manifest = load_action_manifest(action_manifest)
    sft_data_dir = Path(contract["artifacts"]["sft_data"])
    build_result = _build_sft_rows_for_contract(
        contract=contract,
        source=source,
        action_manifest=manifest,
        out_dir=sft_data_dir,
    )
    build_wandb_mode = log_sft_build_to_wandb(
        output_dir=sft_data_dir,
        metrics={
            "accepted_policy_rows": build_result.accepted_policy_rows,
            "accepted_next_frame_rows": build_result.accepted_next_frame_rows,
            "rejected_rows": build_result.rejected_rows,
        },
        config={
            "run_id": run_id,
            "dataset_name": source.dataset_name,
            "dataset_source": contract["dataset"]["source"],
            "db": source.effective_db,
            "source_db": db,
            "nle_root": nle_root,
            "archive_manifest": source.archive_manifest,
            "archive_shard_count": source.archive_shard_count,
            "archive_shard_index": source.archive_shard_index,
            "mode": mode,
            "tasks": list(contract["dataset"]["tasks"]),
            "label_source": contract["dataset"]["label_source"],
            "full_dataset": full_dataset,
            "max_rows": contract["dataset"]["max_rows"],
            "batch_size": batch_size,
            "seq_length": seq_length,
            "seed": seed,
        },
        env=os.environ,
        run_name=f"{run_id}-sft-data-build",
    )
    report = {
        "schema_version": "learn-nethack.sft-build-report.v1",
        "run_id": run_id,
        "status": "completed",
        "contract_path": str(contract_path),
        "db_copy_report": source.db_copy_report,
        "archive_manifest": source.archive_manifest,
        "archive_shard_count": source.archive_shard_count,
        "archive_shard_index": source.archive_shard_index,
        "wandb_mode": wandb_mode.mode,
        "build_wandb_mode": build_wandb_mode,
        "build_result": build_result.__dict__,
    }
    report_path = _write_json(contract["artifacts"]["report"], report)
    return {**report, "report_path": str(report_path)}


def _sft_merge_shards_impl(
    *,
    run_id: str,
    shard_run_ids: str | Sequence[str],
    dry_run_contract: bool = False,
) -> dict[str, Any]:
    contract = local_sft_merge_shards_contract(
        run_id=run_id,
        shard_run_ids=shard_run_ids,
    )
    contract_path = _write_json(contract["artifacts"]["contract"], contract)
    if dry_run_contract:
        return {
            **contract,
            "status": "contract_written",
            "contract_path": str(contract_path),
        }

    wandb_mode = resolve_wandb_mode(os.environ)
    shard_dirs = [Path(path) for path in contract["shard_dataset_dirs"]]
    sft_data_dir = Path(contract["artifacts"]["sft_data"])
    merge_result = merge_sft_dataset_shards(
        shard_dirs=shard_dirs,
        out_dir=sft_data_dir,
    )
    merge_wandb_mode = log_sft_build_to_wandb(
        output_dir=sft_data_dir,
        metrics={
            "accepted_policy_rows": merge_result.accepted_policy_rows,
            "accepted_next_frame_rows": merge_result.accepted_next_frame_rows,
            "rejected_rows": merge_result.rejected_rows,
            "shard_count": len(shard_dirs),
        },
        config={
            "run_id": run_id,
            "source": "sft_shard_merge",
            "shard_run_ids": list(contract["shard_run_ids"]),
            "shard_dataset_dirs": list(contract["shard_dataset_dirs"]),
        },
        env=os.environ,
        run_name=f"{run_id}-sft-shard-merge",
    )
    runs_committed = _commit_mounted_volume("/runs")
    report = {
        "schema_version": "learn-nethack.sft-merge-shards-report.v1",
        "run_id": run_id,
        "status": "completed",
        "contract_path": str(contract_path),
        "wandb_mode": wandb_mode.mode,
        "merge_wandb_mode": merge_wandb_mode,
        "modal_commits": {"runs": runs_committed},
        "merge_result": merge_result.__dict__,
        "shard_run_ids": list(contract["shard_run_ids"]),
        "shard_dataset_dirs": list(contract["shard_dataset_dirs"]),
    }
    report_path = _write_json(contract["artifacts"]["report"], report)
    return {**report, "report_path": str(report_path)}


def _build_sft_rows_for_contract(
    *,
    contract: dict[str, Any],
    source: DecodedBatchSource,
    action_manifest: Any,
    out_dir: Path,
) -> Any:
    task_names = tuple(contract["dataset"]["tasks"])
    label_source = str(contract["dataset"]["label_source"])
    max_rows = contract["dataset"]["max_rows"]
    seed = int(contract["dataset"]["seed"])
    mode = str(contract["dataset"]["mode"])
    progress_callback = _modal_sft_build_progress_logger(
        run_id=str(contract["run_id"]),
        label_source=label_source,
        tasks=task_names,
        progress_path=Path(contract["artifacts"]["report"]).with_name(
            "sft_build_progress.jsonl"
        ),
    )
    if label_source == "pseudo_visible_player_delta":
        return build_pseudo_label_sft_from_frame_batches(
            dataset_name=source.dataset_name,
            mode=mode,
            batches=source.iter_batches(None),
            action_manifest=action_manifest,
            gameids=source.gameids,
            game_metadata_by_id=source.game_metadata_by_id,
            out_dir=out_dir,
            max_rows=max_rows,
            seed=seed,
            tasks=task_names,
            progress_callback=progress_callback,
        )
    return build_sft_from_decoded_batches(
        dataset_name=source.dataset_name,
        mode=mode,
        batches=source.iter_batches(None),
        action_manifest=action_manifest,
        gameids=source.gameids,
        game_metadata_by_id=source.game_metadata_by_id,
        out_dir=out_dir,
        max_rows=max_rows,
        seed=seed,
        tasks=task_names,
        progress_callback=progress_callback,
    )


def _modal_sft_build_progress_logger(
    *,
    run_id: str,
    label_source: str,
    tasks: tuple[str, ...],
    progress_path: Path | None = None,
) -> Callable[[SftBuildProgress], None]:
    def _log(progress: SftBuildProgress) -> None:
        payload = {
            "schema_version": "learn-nethack.sft-build-progress.v1",
            "run_id": run_id,
            "label_source": label_source,
            "tasks": list(tasks),
            **progress.__dict__,
        }
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            mount_path = _modal_volume_mount_for_path(progress_path)
            if mount_path is not None:
                _commit_mounted_volume(mount_path)

    return _log


def _modal_volume_mount_for_path(path: Path) -> str | None:
    path_text = str(path)
    matching_mounts = sorted(
        (
            mount_path
            for mount_path in volumes
            if path_text == mount_path or path_text.startswith(f"{mount_path}/")
        ),
        key=len,
        reverse=True,
    )
    return matching_mounts[0] if matching_mounts else None


def _modal_sft_eval_progress_logger(
    *,
    run_id: str,
    label_source: str,
    tasks: tuple[str, ...],
    progress_path: Path | None = None,
) -> Callable[[dict[str, Any]], None]:
    def _log(progress: dict[str, Any]) -> None:
        payload = {
            "schema_version": "learn-nethack.sft-eval-progress.v1",
            "run_id": run_id,
            "label_source": label_source,
            "tasks": list(tasks),
            **progress,
        }
        line = json.dumps(payload, sort_keys=True)
        print(line, flush=True)
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            mount_path = _modal_volume_mount_for_path(progress_path)
            if mount_path is not None:
                _commit_mounted_volume(mount_path)

    return _log


def _sft_eval_impl(
    *,
    run_id: str,
    db: str | None,
    action_manifest: str,
    nle_root: str | None = None,
    archive_manifest: str | None = None,
    split: str = "validation",
    model_role: str = "baseline",
    adapter: str | None = None,
    model_name: str = SftTrainConfig().model_name,
    mode: str = "single_frame",
    max_rows: int = 128,
    batch_size: int = 4,
    seq_length: int = 128,
    seed: int = 20260615,
    device: str | None = None,
    eval_tasks: str = "policy_action,next_frame",
    label_source: str = "true_keypress",
    next_frame_eval_mode: str = "teacher_forced",
    next_frame_max_new_tokens: int = DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
    next_frame_generate_max_rows: int = 64,
    next_frame_sequence_horizons: str = "1,5,10",
    next_frame_sequence_max_windows: int = 64,
    dry_run_contract: bool = False,
) -> dict[str, Any]:
    normalize_hf_token_env(os.environ)
    contract = local_sft_eval_contract(
        run_id=run_id,
        db=db,
        action_manifest=action_manifest,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
        split=split,
        model_role=model_role,
        adapter=adapter,
        model_name=model_name,
        mode=mode,
        max_rows=max_rows,
        batch_size=batch_size,
        seq_length=seq_length,
        seed=seed,
        eval_tasks=eval_tasks,
        label_source=label_source,
        next_frame_eval_mode=next_frame_eval_mode,
        next_frame_max_new_tokens=next_frame_max_new_tokens,
        next_frame_generate_max_rows=next_frame_generate_max_rows,
        next_frame_sequence_horizons=next_frame_sequence_horizons,
        next_frame_sequence_max_windows=next_frame_sequence_max_windows,
    )
    contract_path = _write_json(contract["artifacts"]["contract"], contract)
    if dry_run_contract:
        return {
            **contract,
            "status": "contract_written",
            "contract_path": str(contract_path),
        }

    wandb_mode = resolve_wandb_mode(os.environ)
    manifest = load_action_manifest(action_manifest)
    source = _decoded_batch_source(
        db=db,
        nle_root=nle_root,
        archive_manifest=archive_manifest,
        batch_size=batch_size,
        seq_length=seq_length,
        staged_db_path=Path(contract["artifacts"]["root"]) / "staged_ttyrecs.db",
    )
    splits = split_gameids(source.gameids, seed=seed)
    split_gameids_by_name = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    if split not in split_gameids_by_name:
        raise ValueError(f"unknown eval split: {split}")
    selected_gameids = split_gameids_by_name[split]
    if not selected_gameids:
        raise RuntimeError(f"eval split {split!r} has no gameids")

    eval_data_dir = Path(contract["artifacts"]["root"]) / "eval-data"
    split_sets = {
        "train": set[int](),
        "validation": set[int](),
        "test": set[int](),
    }
    split_sets[split] = set(selected_gameids)
    batches = source.iter_batches(selected_gameids)
    task_names = tuple(contract["evaluation"]["tasks"])
    eval_label_source = str(contract["dataset"]["label_source"])
    if eval_label_source == "pseudo_visible_player_delta":
        transitions = (
            transition
            for batch in batches
            for transition in normalize_frame_only_batch(batch)
        )
        build_result = write_pseudo_label_policy_dataset(
            dataset_name=source.dataset_name,
            mode=mode,
            transitions=transitions,
            action_manifest=manifest,
            game_metadata_by_id=source.game_metadata_by_id,
            splits=split_sets,
            out_dir=eval_data_dir,
            max_rows=max_rows,
            tasks=task_names,
        )
    else:
        transitions = (
            transition
            for batch in batches
            for transition in normalize_decoded_batch(batch)
        )
        build_result = write_sft_dataset(
            dataset_name=source.dataset_name,
            mode=mode,
            transitions=transitions,
            action_manifest=manifest,
            game_metadata_by_id=source.game_metadata_by_id,
            splits=split_sets,
            out_dir=eval_data_dir,
            max_rows=max_rows,
            tasks=task_names,
        )
    policy_rows_path = eval_data_dir / f"{split}.policy_action.jsonl"
    next_frame_rows_path = eval_data_dir / f"{split}.next_frame.jsonl"
    policy_rows = (
        load_jsonl_rows([policy_rows_path])
        if "policy_action" in task_names and policy_rows_path.exists()
        else []
    )
    next_frame_rows = (
        load_jsonl_rows([next_frame_rows_path])
        if "next_frame" in task_names and next_frame_rows_path.exists()
        else []
    )
    hf_cache_committed_after_policy_load = False
    policy_metrics: dict[str, float] = {}
    if "policy_action" in task_names:
        policy = TransformerCandidatePolicy(
            ModelWatchSpec(
                role=model_role,
                model_name=model_name,
                adapter_checkpoint=adapter,
            ),
            device=device,
        )
        hf_cache_committed_after_policy_load = _commit_mounted_volume(
            HF_CACHE_MOUNT_PATH
        )
        policy_metrics = evaluate_policy_rows_with_policy(
            rows=policy_rows,
            policy=policy,
            max_rows=max_rows,
        )
    hf_cache_committed_after_dynamics_load = False
    next_frame_metrics: dict[str, float] = {}
    next_frame_generated_samples: list[dict[str, Any]] = []
    if "next_frame" in task_names:
        frame_eval_mode = str(contract["evaluation"]["next_frame_eval_mode"])
        sequence_horizons = tuple(
            int(value)
            for value in contract["evaluation"]["next_frame_sequence_horizons"]
        )
        next_frame_metrics.update(
            summarize_next_frame_sequence_rows(
                rows=next_frame_rows,
                horizons=sequence_horizons,
            )
        )
        eval_progress_callback = _modal_sft_eval_progress_logger(
            run_id=run_id,
            label_source=eval_label_source,
            tasks=task_names,
            progress_path=Path(contract["artifacts"]["progress"]),
        )
        eval_sample_callback = _bounded_sample_collector(
            next_frame_generated_samples,
            max_samples=8,
        )
        dynamics_predictor = TransformerNextFramePredictor(
            DynamicsModelSpec(
                model_name=model_name,
                adapter_checkpoint=adapter,
            ),
            device=device,
            max_new_tokens=next_frame_max_new_tokens,
        )
        hf_cache_committed_after_dynamics_load = _commit_mounted_volume(
            HF_CACHE_MOUNT_PATH
        )
        if frame_eval_mode in {"teacher_forced", "both"}:
            next_frame_metrics.update(
                evaluate_next_frame_rows_with_scorer(
                    rows=next_frame_rows,
                    scorer=dynamics_predictor,
                    max_rows=max_rows,
                )
            )
        if frame_eval_mode in {"generate", "both"}:
            next_frame_metrics.update(
                evaluate_next_frame_rows_with_predictor(
                    rows=next_frame_rows,
                    predictor=dynamics_predictor,
                    max_rows=int(
                        contract["evaluation"]["next_frame_generate_max_rows"]
                    ),
                    progress_callback=eval_progress_callback,
                    sample_callback=eval_sample_callback,
                )
            )
            next_frame_metrics.update(
                evaluate_next_frame_sequences_with_predictor(
                    rows=next_frame_rows,
                    predictor=dynamics_predictor,
                    horizons=sequence_horizons,
                    max_windows=int(
                        contract["evaluation"]["next_frame_sequence_max_windows"]
                    ),
                    progress_callback=eval_progress_callback,
                    sample_callback=eval_sample_callback,
                )
            )
    metrics = {**policy_metrics, **next_frame_metrics}
    metrics_path = _write_json(contract["artifacts"]["metrics"], metrics)
    report = {
        "schema_version": "learn-nethack.sft-eval-report.v1",
        "run_id": run_id,
        "status": "completed",
        "contract_path": str(contract_path),
        "metrics_path": str(metrics_path),
        "db_copy_report": source.db_copy_report,
        "archive_manifest": source.archive_manifest,
        "archive_shard_count": source.archive_shard_count,
        "wandb_mode": wandb_mode.mode,
        "dataset": {
            "source": contract["dataset"]["source"],
            "db": source.effective_db,
            "dataset_name": source.dataset_name,
            "split": split,
            "selected_gameids": len(selected_gameids),
            "policy_rows_path": str(policy_rows_path) if policy_rows else None,
            "next_frame_rows_path": (
                str(next_frame_rows_path) if next_frame_rows else None
            ),
            "max_rows": max_rows,
            "label_source": eval_label_source,
        },
        "evaluation": contract["evaluation"],
        "model": contract["model"],
        "build_result": build_result.__dict__,
        "metrics": metrics,
        "next_frame_generated_samples": next_frame_generated_samples,
        "hf_cache": {
            "mount_path": HF_CACHE_MOUNT_PATH,
            "committed_after_policy_load": hf_cache_committed_after_policy_load,
            "committed_after_dynamics_load": hf_cache_committed_after_dynamics_load,
        },
    }
    report_path = _write_json(contract["artifacts"]["report"], report)

    import wandb

    run = wandb.init(
        project="learn-nethack",
        name=run_id,
        job_type="sft-eval",
        mode=wandb_mode.mode,
        config=report,
        dir=contract["artifacts"]["wandb"],
    )
    report["wandb"] = _wandb_run_report(run, wandb_mode.mode)
    report_path = _write_json(contract["artifacts"]["report"], report)
    run.log({f"sft_eval/{key}": value for key, value in sorted(metrics.items())})
    artifact = wandb.Artifact(name=f"sft-eval-{run_id}", type="evaluation")
    artifact.add_file(str(metrics_path))
    artifact.add_file(str(report_path))
    if policy_rows_path.exists():
        artifact.add_file(str(policy_rows_path))
    if next_frame_rows_path.exists():
        artifact.add_file(str(next_frame_rows_path))
    run.log_artifact(artifact)
    run.finish()
    return {**report, "report_path": str(report_path)}


def _bounded_sample_collector(
    samples: list[dict[str, Any]],
    *,
    max_samples: int,
) -> Callable[[dict[str, Any]], None]:
    def _collect(sample: dict[str, Any]) -> None:
        if len(samples) >= max_samples:
            return
        samples.append(dict(sample))

    return _collect


def _sft_compare_impl(
    *,
    baseline: str,
    trained: str,
    out: str,
    trained_run_id: str,
    baseline_run_id: str | None = None,
) -> dict[str, Any]:
    baseline_metrics = json.loads(Path(baseline).read_text(encoding="utf-8"))
    trained_metrics = json.loads(Path(trained).read_text(encoding="utf-8"))
    report = build_score_to_beat_report(
        baseline_metrics=baseline_metrics,
        trained_metrics=trained_metrics,
        baseline_run_id=baseline_run_id,
        trained_run_id=trained_run_id,
    )
    report_path = _write_json(out, report)
    return {**report, "report_path": str(report_path)}


def _watch_compare_impl(
    *,
    run_id: str,
    action_manifest: str,
    current_checkpoint: str | None = None,
    model_name: str = DEFAULT_WATCH_MODEL_NAME,
    env_id: str = DEFAULT_WATCH_ENV_ID,
    character: str = DEFAULT_NLE_CHARACTER,
    seed: int = 20260615,
    max_steps: int = 10,
    device: str | None = None,
    dry_run_contract: bool = False,
) -> dict[str, Any]:
    normalize_hf_token_env(os.environ)
    contract = local_watch_compare_contract(
        run_id=run_id,
        action_manifest=action_manifest,
        current_checkpoint=current_checkpoint,
        model_name=model_name,
        env_id=env_id,
        character=character,
        seed=seed,
        max_steps=max_steps,
        device=device,
    )
    contract_path = _write_json(contract["artifacts"]["contract"], contract)
    if dry_run_contract:
        return {
            **contract,
            "status": "contract_written",
            "contract_path": str(contract_path),
        }

    wandb_mode = resolve_wandb_mode(os.environ)
    report = run_checkpoint_compare(
        run_id=run_id,
        current_checkpoint=contract["watch"]["current_checkpoint"],
        action_manifest_path=action_manifest,
        out_dir=contract["artifacts"]["watch_dir"],
        model_name=model_name,
        env_id=env_id,
        character=character,
        seed=seed,
        max_steps=max_steps,
        device=device,
    )
    events_path = Path(report["events_path"])
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hf_cache_committed = _commit_mounted_volume(HF_CACHE_MOUNT_PATH)
    watch_committed = _commit_mounted_volume("/watch")
    runs_committed = _commit_mounted_volume("/runs")
    current_last = events[-1]["current"] if events else {}
    baseline_last = events[-1]["baseline"] if events else {}
    rollout_metrics = report.get("rollout_metrics", {})
    current_rollout_metrics = dict(rollout_metrics.get("current") or {})
    baseline_rollout_metrics = dict(rollout_metrics.get("baseline") or {})
    rollout_deltas = dict(rollout_metrics.get("deltas") or {})
    watch_report = {
        **report,
        "contract_path": str(contract_path),
        "events": events,
        "modal_commits": {
            "hf_cache": hf_cache_committed,
            "watch": watch_committed,
            "runs": runs_committed,
        },
        "wandb_mode": wandb_mode.mode,
    }

    import wandb

    run = wandb.init(
        project="learn-nethack",
        name=run_id,
        job_type="watch-compare",
        mode=wandb_mode.mode,
        config=watch_report,
        dir=contract["artifacts"]["root"],
    )
    watch_report["wandb"] = _wandb_run_report(run, wandb_mode.mode)
    _write_json(contract["artifacts"]["report"], watch_report)
    watch_metric_names = (
        "fitness_score",
        "cumulative_reward",
        "score_delta",
        "depth_delta",
        "depth_max",
        "hp_damage_observed",
        "wall_message_count",
        "wall_message_rate",
        "bad_message_count",
        "bad_message_rate",
        "non_advancing_step_count",
        "non_advancing_step_rate",
        "action_repeat_rate",
        "action_collapse_excess",
        "action_collapse_rate_excess",
        "visible_map_novelty_count",
        "visible_map_novelty_bonus_count",
        "meaningful_event_count",
        "meaningful_event_bonus_count",
        "raw_live_progress_event_count",
        "clean_live_progress_event_count",
        "live_progress_event_count",
        "dirty_live_progress_event_count",
        "hunger_warning_count",
        "starvation_or_faint_count",
        "menu_or_prompt_step_count",
        "menu_or_prompt_step_rate",
        "stuck_menu_or_prompt_loop_count",
        "zero_progress_episode",
    )
    watch_log = {
        "watch/event_count": len(events),
        "watch/current_done": float(bool(current_last.get("done", False))),
        "watch/baseline_done": float(bool(baseline_last.get("done", False))),
    }
    for side_name, metrics in (
        ("current", current_rollout_metrics),
        ("baseline", baseline_rollout_metrics),
    ):
        for metric_name in watch_metric_names:
            watch_log[f"watch/{side_name}_{metric_name}"] = float(
                metrics.get(metric_name) or 0.0
            )
    for metric_name in watch_metric_names:
        watch_log[f"watch/delta_{metric_name}"] = float(
            rollout_deltas.get(metric_name) or 0.0
        )
    run.log(watch_log)
    artifact = wandb.Artifact(name=f"watch-compare-{run_id}", type="evaluation")
    for path_name in ("events_path", "latest_path", "viewer_path"):
        artifact.add_file(str(report[path_name]))
    artifact.add_file(str(Path(contract["artifacts"]["report"])))
    artifact.add_file(str(contract_path))
    run.log_artifact(artifact)
    run.finish()
    return {
        **watch_report,
    }


def _watch_compare_sweep_impl(
    *,
    run_id: str,
    action_manifest: str,
    current_checkpoint: str | None = None,
    model_name: str = DEFAULT_WATCH_MODEL_NAME,
    env_id: str = DEFAULT_WATCH_ENV_ID,
    character: str = DEFAULT_NLE_CHARACTER,
    seeds: str = "20260615,20260616,20260617",
    max_steps: int = 10,
    device: str | None = None,
    dry_run_contract: bool = False,
) -> dict[str, Any]:
    normalize_hf_token_env(os.environ)
    contract = local_watch_compare_sweep_contract(
        run_id=run_id,
        action_manifest=action_manifest,
        current_checkpoint=current_checkpoint,
        model_name=model_name,
        env_id=env_id,
        character=character,
        seeds=seeds,
        max_steps=max_steps,
        device=device,
    )
    contract_path = _write_json(contract["artifacts"]["contract"], contract)
    if dry_run_contract:
        return {
            **contract,
            "status": "contract_written",
            "contract_path": str(contract_path),
        }

    wandb_mode = resolve_wandb_mode(os.environ)
    report = run_checkpoint_compare_sweep(
        run_id=run_id,
        current_checkpoint=contract["watch"]["current_checkpoint"],
        action_manifest_path=action_manifest,
        out_dir=contract["artifacts"]["watch_dir"],
        model_name=model_name,
        env_id=env_id,
        character=character,
        seeds=contract["watch"]["seeds"],
        max_steps=max_steps,
        device=device,
    )
    hf_cache_committed = _commit_mounted_volume(HF_CACHE_MOUNT_PATH)
    watch_committed = _commit_mounted_volume("/watch")
    runs_committed = _commit_mounted_volume("/runs")
    watch_report = {
        **report,
        "contract_path": str(contract_path),
        "modal_commits": {
            "hf_cache": hf_cache_committed,
            "watch": watch_committed,
            "runs": runs_committed,
        },
        "wandb_mode": wandb_mode.mode,
    }

    import wandb

    run = wandb.init(
        project="learn-nethack",
        name=run_id,
        job_type="watch-compare-sweep",
        mode=wandb_mode.mode,
        config=watch_report,
        dir=contract["artifacts"]["root"],
    )
    watch_report["wandb"] = _wandb_run_report(run, wandb_mode.mode)
    _write_json(contract["artifacts"]["report"], watch_report)
    run.log(_watch_sweep_wandb_metrics(watch_report))
    artifact = wandb.Artifact(name=f"watch-compare-sweep-{run_id}", type="evaluation")
    artifact.add_file(str(contract["artifacts"]["report"]))
    artifact.add_file(str(contract_path))
    artifact_names: set[str] = set()
    for path, artifact_name in _watch_sweep_artifact_file_entries(
        watch_report.get("seed_reports", [])
    ):
        if artifact_name in artifact_names:
            raise ValueError(f"duplicate watch sweep artifact path: {artifact_name}")
        artifact_names.add(artifact_name)
        artifact.add_file(str(path), name=artifact_name)
    run.log_artifact(artifact)
    run.finish()
    return watch_report


def _watch_sweep_artifact_file_entries(
    seed_reports: Any,
) -> list[tuple[Path, str]]:
    if not isinstance(seed_reports, Sequence) or isinstance(seed_reports, (str, bytes)):
        return []
    entries: list[tuple[Path, str]] = []
    for seed_report in seed_reports:
        if not isinstance(seed_report, Mapping):
            continue
        seed = seed_report.get("seed", "unknown")
        seed_dir_name = f"seed-{seed}"
        for key in ("events_path", "latest_path", "viewer_path", "report_path"):
            path = seed_report.get(key)
            if not path:
                continue
            local_path = Path(str(path))
            if local_path.exists():
                entries.append((local_path, f"{seed_dir_name}/{local_path.name}"))
    return entries


def _watch_sweep_wandb_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "watch_sweep/seed_count": float(report.get("seed_count", 0) or 0),
        "watch_sweep/paired_initial_state_equal_count": float(
            report.get("paired_initial_state_equal_count", 0) or 0
        ),
        "watch_sweep/deterministic_nle_seed_count": float(
            report.get("deterministic_nle_seed_count", 0) or 0
        ),
        "watch_sweep/total_event_count": float(report.get("total_event_count", 0) or 0),
    }
    rollout_metrics = report.get("rollout_metrics")
    if not isinstance(rollout_metrics, Mapping):
        return metrics
    for side_name in ("current", "baseline", "deltas"):
        side_metrics = rollout_metrics.get(side_name)
        if not isinstance(side_metrics, Mapping):
            continue
        for metric_name, value in side_metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                metrics[f"watch_sweep/{side_name}_{metric_name}"] = float(value)
    return metrics


def _extract_nld_shard_impl(
    *,
    shard: str,
    destination: str = "/datasets/nld-nao-unzipped",
    report: str | None = None,
    commit_volume: bool = False,
) -> dict[str, Any]:
    extract_report = safe_extract_tar_shard(
        shard_path=shard,
        destination_root=destination,
    )
    should_commit = _commit_mounted_volume("/datasets") if commit_volume else False
    report_payload = {
        "schema_version": "learn-nethack.nld-shard-modal-extract.v1",
        "status": "completed",
        "shard": shard,
        "destination": destination,
        "committed": should_commit,
        "extract": extract_report,
    }
    report_path = report or (f"/runs/nld-shard-extract/reports/{Path(shard).name}.json")
    written_report = _write_json(report_path, report_payload)
    return {**report_payload, "report_path": str(written_report)}


def _readiness_impl(run_id: str) -> dict:
    report = local_readiness_report(run_id)
    report["execution"] = modal_cloud_execution_context(
        function_call_id=modal.current_function_call_id()
        if modal is not None
        else None,
        input_id=modal.current_input_id() if modal is not None else None,
        env=os.environ,
        hostname=socket.gethostname(),
    )
    report_path = _write_report(report)

    import wandb

    wandb_run = wandb.init(
        project="learn-nethack",
        job_type="modal-readiness",
        name=run_id,
        config=report,
        dir=report["artifact_layout"]["wandb"],
    )
    artifact_name = f"modal-readiness-{run_id}"
    report["wandb"].update(_wandb_run_report(wandb_run, report["wandb"]["mode"]))
    report["wandb"]["artifact_name"] = artifact_name
    report["hf_cache"] = {
        "mount_path": HF_CACHE_MOUNT_PATH,
        "volume_name": modal_volume_mounts()[HF_CACHE_MOUNT_PATH],
    }
    report_path = _write_report(report)
    wandb_run.log({"modal/readiness": 1})
    readiness_artifact = wandb.Artifact(name=artifact_name, type="evaluation")
    readiness_artifact.add_file(str(report_path))
    wandb_run.log_artifact(readiness_artifact)
    wandb_run.finish()
    _commit_mounted_volume("/runs")

    report["report_path"] = str(report_path)
    return report


if app is not None:

    @app.function(
        image=image,
        env=function_env,
        gpu=DEFAULT_GPU,
        volumes=volumes,
        secrets=secrets,
        timeout=20 * 60,
    )
    def readiness(run_id: str = "modal-readiness-smoke") -> dict:
        """Verify Modal image, volume, secret, artifact, and W&B readiness."""
        return _readiness_impl(run_id)

    readiness = _make_modal_function_callable(readiness)

    @app.function(
        image=image,
        env=function_env,
        volumes=volumes,
        secrets=secrets,
        timeout=12 * 60 * 60,
    )
    def sft_build(
        run_id: str,
        action_manifest: str,
        db: str | None = None,
        nle_root: str | None = None,
        archive_manifest: str | None = None,
        archive_shard_index: int | None = None,
        mode: str = "single_frame",
        full_dataset: bool = False,
        max_rows: int = 1000,
        batch_size: int = 4,
        seq_length: int = 128,
        seed: int = 20260615,
        tasks: str = "policy_action,next_frame",
        label_source: str = "true_keypress",
        dry_run_contract: bool = False,
    ) -> dict[str, Any]:
        """Build SFT rows and reports without loading trainer state."""
        return _sft_build_impl(
            run_id=run_id,
            db=db,
            action_manifest=action_manifest,
            nle_root=nle_root,
            archive_manifest=archive_manifest,
            archive_shard_index=archive_shard_index,
            mode=mode,
            full_dataset=full_dataset,
            max_rows=max_rows,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            tasks=tasks,
            label_source=label_source,
            dry_run_contract=dry_run_contract,
        )

    sft_build = _make_modal_function_callable(sft_build)

    @app.function(
        image=image,
        env=function_env,
        volumes=volumes,
        secrets=secrets,
        timeout=6 * 60 * 60,
    )
    def sft_merge_shards(
        run_id: str,
        shard_run_ids: str,
        dry_run_contract: bool = False,
    ) -> dict[str, Any]:
        """Merge completed SFT shard datasets into one trainable dataset."""
        return _sft_merge_shards_impl(
            run_id=run_id,
            shard_run_ids=shard_run_ids,
            dry_run_contract=dry_run_contract,
        )

    sft_merge_shards = _make_modal_function_callable(sft_merge_shards)

    @app.function(
        image=image,
        env=function_env,
        gpu=DEFAULT_GPU,
        volumes=volumes,
        secrets=secrets,
        timeout=24 * 60 * 60,
    )
    def sft_train(
        run_id: str,
        action_manifest: str,
        db: str | None = None,
        nle_root: str | None = None,
        archive_manifest: str | None = None,
        mode: str = "single_frame",
        full_dataset: bool = False,
        max_rows: int = 1000,
        batch_size: int = 4,
        seq_length: int = 128,
        seed: int = 20260615,
        tasks: str = "policy_action,next_frame",
        label_source: str = "true_keypress",
        model_name: str = SftTrainConfig().model_name,
        max_steps: int = SftTrainConfig().max_steps,
        dry_run_contract: bool = False,
    ) -> dict[str, Any]:
        """Run or contract-check the Modal Unsloth SFT sequence."""
        return _sft_train_impl(
            run_id=run_id,
            db=db,
            action_manifest=action_manifest,
            nle_root=nle_root,
            archive_manifest=archive_manifest,
            mode=mode,
            full_dataset=full_dataset,
            max_rows=max_rows,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            tasks=tasks,
            label_source=label_source,
            model_name=model_name,
            max_steps=max_steps,
            dry_run_contract=dry_run_contract,
        )

    sft_train = _make_modal_function_callable(sft_train)

    @app.function(
        image=image,
        env=function_env,
        gpu=DEFAULT_GPU,
        volumes=volumes,
        secrets=secrets,
        timeout=24 * 60 * 60,
    )
    def sft_train_existing(
        run_id: str,
        dataset_dir: str,
        model_name: str = SftTrainConfig().model_name,
        max_steps: int = SftTrainConfig().max_steps,
        training_objective: str = SftTrainConfig().training_objective,
        dry_run_contract: bool = False,
    ) -> dict[str, Any]:
        """Train on an existing SFT JSONL dataset in a Modal volume."""
        return _sft_train_existing_impl(
            run_id=run_id,
            dataset_dir=dataset_dir,
            model_name=model_name,
            max_steps=max_steps,
            training_objective=training_objective,
            dry_run_contract=dry_run_contract,
        )

    sft_train_existing = _make_modal_function_callable(sft_train_existing)

    @app.function(
        image=image,
        env=function_env,
        gpu=DEFAULT_GPU,
        volumes=volumes,
        secrets=secrets,
        timeout=8 * 60 * 60,
    )
    def sft_eval(
        run_id: str,
        action_manifest: str,
        db: str | None = None,
        nle_root: str | None = None,
        archive_manifest: str | None = None,
        split: str = "validation",
        model_role: str = "baseline",
        adapter: str | None = None,
        model_name: str = SftTrainConfig().model_name,
        mode: str = "single_frame",
        max_rows: int = 128,
        batch_size: int = 4,
        seq_length: int = 128,
        seed: int = 20260615,
        device: str | None = None,
        eval_tasks: str = "policy_action,next_frame",
        label_source: str = "true_keypress",
        next_frame_eval_mode: str = "teacher_forced",
        next_frame_max_new_tokens: int = DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
        next_frame_generate_max_rows: int = 64,
        next_frame_sequence_horizons: str = "1,5,10",
        next_frame_sequence_max_windows: int = 64,
        dry_run_contract: bool = False,
    ) -> dict[str, Any]:
        """Run or contract-check baseline/trained SFT evaluation."""
        return _sft_eval_impl(
            run_id=run_id,
            db=db,
            action_manifest=action_manifest,
            nle_root=nle_root,
            archive_manifest=archive_manifest,
            split=split,
            model_role=model_role,
            adapter=adapter,
            model_name=model_name,
            mode=mode,
            max_rows=max_rows,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            device=device,
            eval_tasks=eval_tasks,
            label_source=label_source,
            next_frame_eval_mode=next_frame_eval_mode,
            next_frame_max_new_tokens=next_frame_max_new_tokens,
            next_frame_generate_max_rows=next_frame_generate_max_rows,
            next_frame_sequence_horizons=next_frame_sequence_horizons,
            next_frame_sequence_max_windows=next_frame_sequence_max_windows,
            dry_run_contract=dry_run_contract,
        )

    sft_eval = _make_modal_function_callable(sft_eval)

    @app.function(
        image=image,
        env=function_env,
        gpu=DEFAULT_GPU,
        volumes=volumes,
        secrets=secrets,
        timeout=2 * 60 * 60,
    )
    def watch_compare(
        run_id: str,
        action_manifest: str,
        current_checkpoint: str | None = None,
        model_name: str = DEFAULT_WATCH_MODEL_NAME,
        env_id: str = DEFAULT_WATCH_ENV_ID,
        character: str = DEFAULT_NLE_CHARACTER,
        seed: int = 20260615,
        max_steps: int = 10,
        device: str | None = None,
        dry_run_contract: bool = False,
    ) -> dict[str, Any]:
        """Run a Modal side-by-side watch comparison."""
        return _watch_compare_impl(
            run_id=run_id,
            action_manifest=action_manifest,
            current_checkpoint=current_checkpoint,
            model_name=model_name,
            env_id=env_id,
            character=character,
            seed=seed,
            max_steps=max_steps,
            device=device,
            dry_run_contract=dry_run_contract,
        )

    watch_compare = _make_modal_function_callable(watch_compare)

    @app.function(
        image=image,
        env=function_env,
        gpu=DEFAULT_GPU,
        volumes=volumes,
        secrets=secrets,
        timeout=2 * 60 * 60,
    )
    def watch_compare_sweep(
        run_id: str,
        action_manifest: str,
        current_checkpoint: str | None = None,
        model_name: str = DEFAULT_WATCH_MODEL_NAME,
        env_id: str = DEFAULT_WATCH_ENV_ID,
        character: str = DEFAULT_NLE_CHARACTER,
        seeds: str = "20260615,20260616,20260617",
        max_steps: int = 10,
        device: str | None = None,
        dry_run_contract: bool = False,
    ) -> dict[str, Any]:
        """Run a Modal multi-seed side-by-side watch comparison."""
        return _watch_compare_sweep_impl(
            run_id=run_id,
            action_manifest=action_manifest,
            current_checkpoint=current_checkpoint,
            model_name=model_name,
            env_id=env_id,
            character=character,
            seeds=seeds,
            max_steps=max_steps,
            device=device,
            dry_run_contract=dry_run_contract,
        )

    watch_compare_sweep = _make_modal_function_callable(watch_compare_sweep)

    @app.function(
        image=image,
        env=function_env,
        volumes=volumes,
        secrets=secrets,
        timeout=30 * 60,
    )
    def sft_compare(
        baseline: str,
        trained: str,
        out: str,
        trained_run_id: str,
        baseline_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Compare baseline and trained SFT metrics into the score-to-beat report."""
        return _sft_compare_impl(
            baseline=baseline,
            trained=trained,
            out=out,
            trained_run_id=trained_run_id,
            baseline_run_id=baseline_run_id,
        )

    sft_compare = _make_modal_function_callable(sft_compare)

    @app.function(
        image=image,
        env=function_env,
        volumes=volumes,
        timeout=12 * 60 * 60,
    )
    def extract_nld_shard(
        shard: str,
        destination: str = "/datasets/nld-nao-unzipped",
        report: str | None = None,
    ) -> dict[str, Any]:
        """Extract a tar shard into the datasets volume and commit it."""
        return _extract_nld_shard_impl(
            shard=shard,
            destination=destination,
            report=report,
            commit_volume=True,
        )

    extract_nld_shard = _make_modal_function_callable(extract_nld_shard)

else:

    def readiness(run_id: str = "modal-readiness-smoke") -> dict:
        """Local fallback for static checks when Modal is not installed."""
        return local_readiness_report(run_id)

    def sft_build(
        run_id: str,
        action_manifest: str,
        db: str | None = None,
        nle_root: str | None = None,
        archive_manifest: str | None = None,
        archive_shard_index: int | None = None,
        mode: str = "single_frame",
        full_dataset: bool = False,
        max_rows: int = 1000,
        batch_size: int = 4,
        seq_length: int = 128,
        seed: int = 20260615,
        tasks: str = "policy_action,next_frame",
        label_source: str = "true_keypress",
        dry_run_contract: bool = True,
    ) -> dict[str, Any]:
        """Local fallback returns or executes the durable SFT build contract."""
        return _sft_build_impl(
            run_id=run_id,
            db=db,
            action_manifest=action_manifest,
            nle_root=nle_root,
            archive_manifest=archive_manifest,
            archive_shard_index=archive_shard_index,
            mode=mode,
            full_dataset=full_dataset,
            max_rows=max_rows,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            tasks=tasks,
            label_source=label_source,
            dry_run_contract=dry_run_contract,
        )

    def sft_merge_shards(
        run_id: str,
        shard_run_ids: str,
        dry_run_contract: bool = True,
    ) -> dict[str, Any]:
        """Local fallback returns or executes the shard merge contract."""
        return _sft_merge_shards_impl(
            run_id=run_id,
            shard_run_ids=shard_run_ids,
            dry_run_contract=dry_run_contract,
        )

    def sft_train(
        run_id: str,
        action_manifest: str,
        db: str | None = None,
        nle_root: str | None = None,
        archive_manifest: str | None = None,
        mode: str = "single_frame",
        full_dataset: bool = False,
        max_rows: int = 1000,
        batch_size: int = 4,
        seq_length: int = 128,
        seed: int = 20260615,
        tasks: str = "policy_action,next_frame",
        label_source: str = "true_keypress",
        model_name: str = SftTrainConfig().model_name,
        max_steps: int = SftTrainConfig().max_steps,
        dry_run_contract: bool = True,
    ) -> dict[str, Any]:
        """Local fallback returns the durable SFT contract."""
        return _sft_train_impl(
            run_id=run_id,
            db=db,
            action_manifest=action_manifest,
            nle_root=nle_root,
            archive_manifest=archive_manifest,
            mode=mode,
            full_dataset=full_dataset,
            max_rows=max_rows,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            tasks=tasks,
            label_source=label_source,
            model_name=model_name,
            max_steps=max_steps,
            dry_run_contract=dry_run_contract,
        )

    def sft_train_existing(
        run_id: str,
        dataset_dir: str,
        model_name: str = SftTrainConfig().model_name,
        max_steps: int = SftTrainConfig().max_steps,
        training_objective: str = SftTrainConfig().training_objective,
        dry_run_contract: bool = True,
    ) -> dict[str, Any]:
        """Local fallback returns the existing-dataset SFT contract."""
        return _sft_train_existing_impl(
            run_id=run_id,
            dataset_dir=dataset_dir,
            model_name=model_name,
            max_steps=max_steps,
            training_objective=training_objective,
            dry_run_contract=dry_run_contract,
        )

    def sft_eval(
        run_id: str,
        action_manifest: str,
        db: str | None = None,
        nle_root: str | None = None,
        archive_manifest: str | None = None,
        split: str = "validation",
        model_role: str = "baseline",
        adapter: str | None = None,
        model_name: str = SftTrainConfig().model_name,
        mode: str = "single_frame",
        max_rows: int = 128,
        batch_size: int = 4,
        seq_length: int = 128,
        seed: int = 20260615,
        device: str | None = None,
        eval_tasks: str = "policy_action,next_frame",
        label_source: str = "true_keypress",
        next_frame_eval_mode: str = "teacher_forced",
        next_frame_max_new_tokens: int = DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
        next_frame_generate_max_rows: int = 64,
        next_frame_sequence_horizons: str = "1,5,10",
        next_frame_sequence_max_windows: int = 64,
        dry_run_contract: bool = True,
    ) -> dict[str, Any]:
        """Local fallback returns the durable eval contract."""
        return _sft_eval_impl(
            run_id=run_id,
            db=db,
            action_manifest=action_manifest,
            nle_root=nle_root,
            archive_manifest=archive_manifest,
            split=split,
            model_role=model_role,
            adapter=adapter,
            model_name=model_name,
            mode=mode,
            max_rows=max_rows,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            device=device,
            eval_tasks=eval_tasks,
            label_source=label_source,
            next_frame_eval_mode=next_frame_eval_mode,
            next_frame_max_new_tokens=next_frame_max_new_tokens,
            next_frame_generate_max_rows=next_frame_generate_max_rows,
            next_frame_sequence_horizons=next_frame_sequence_horizons,
            next_frame_sequence_max_windows=next_frame_sequence_max_windows,
            dry_run_contract=dry_run_contract,
        )

    def watch_compare(
        run_id: str,
        action_manifest: str,
        current_checkpoint: str | None = None,
        model_name: str = DEFAULT_WATCH_MODEL_NAME,
        env_id: str = DEFAULT_WATCH_ENV_ID,
        character: str = DEFAULT_NLE_CHARACTER,
        seed: int = 20260615,
        max_steps: int = 10,
        device: str | None = None,
        dry_run_contract: bool = True,
    ) -> dict[str, Any]:
        """Local fallback returns the durable watch-compare contract."""
        return _watch_compare_impl(
            run_id=run_id,
            action_manifest=action_manifest,
            current_checkpoint=current_checkpoint,
            model_name=model_name,
            env_id=env_id,
            character=character,
            seed=seed,
            max_steps=max_steps,
            device=device,
            dry_run_contract=dry_run_contract,
        )

    def watch_compare_sweep(
        run_id: str,
        action_manifest: str,
        current_checkpoint: str | None = None,
        model_name: str = DEFAULT_WATCH_MODEL_NAME,
        env_id: str = DEFAULT_WATCH_ENV_ID,
        character: str = DEFAULT_NLE_CHARACTER,
        seeds: str = "20260615,20260616,20260617",
        max_steps: int = 10,
        device: str | None = None,
        dry_run_contract: bool = True,
    ) -> dict[str, Any]:
        """Local fallback returns the durable watch-sweep contract."""
        return _watch_compare_sweep_impl(
            run_id=run_id,
            action_manifest=action_manifest,
            current_checkpoint=current_checkpoint,
            model_name=model_name,
            env_id=env_id,
            character=character,
            seeds=seeds,
            max_steps=max_steps,
            device=device,
            dry_run_contract=dry_run_contract,
        )

    def sft_compare(
        baseline: str,
        trained: str,
        out: str,
        trained_run_id: str,
        baseline_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Local fallback compares two metrics files."""
        return _sft_compare_impl(
            baseline=baseline,
            trained=trained,
            out=out,
            trained_run_id=trained_run_id,
            baseline_run_id=baseline_run_id,
        )

    def extract_nld_shard(
        shard: str,
        destination: str = "/datasets/nld-nao-unzipped",
        report: str | None = None,
    ) -> dict[str, Any]:
        """Local fallback extracts a shard without Modal volume commit."""
        return _extract_nld_shard_impl(
            shard=shard,
            destination=destination,
            report=report,
            commit_volume=False,
        )
