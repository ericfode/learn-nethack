"""Static Modal readiness contract for Gemma/NetHack training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping, MutableMapping


MODAL_APP_NAME = "learn-nethack-gemma"
MODAL_SECRET_NAMES = ("hf-token", "wandb-secret")
DEFAULT_GPU = "A100"
PYTHON_VERSION = "3.11"
MODAL_SOURCE_MODULES = ("learn_nethack",)
HF_CACHE_MOUNT_PATH = "/cache/huggingface"
DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS = 512
DEFAULT_WATCH_ENV_ID = "NetHackChallenge-v0"
DEFAULT_WATCH_MODEL_NAME = "google/gemma-4-E4b-it"
DEFAULT_WATCH_ACTION_MANIFEST = "/datasets/action_manifest.json"
DEFAULT_WATCH_PROOF_SEEDS = (
    "20260615,20260616,20260617,20260618,"
    "20260619,20260620,20260621,20260622,"
    "20260623,20260624,20260625,20260626,"
    "20260627,20260628,20260629,20260630"
)


@dataclass(frozen=True)
class ModalVolumeSpec:
    name: str
    mount_path: str
    purpose: str


@dataclass(frozen=True)
class WandbMode:
    mode: str
    requires_api_key: bool


MODAL_VOLUMES = (
    ModalVolumeSpec(
        name="learn-nethack-datasets",
        mount_path="/datasets",
        purpose="SFT JSONL shards, manifests, and validation splits",
    ),
    ModalVolumeSpec(
        name="learn-nethack-runs",
        mount_path="/runs",
        purpose="reports, adapters, replay media, ttyrecs, and W&B run files",
    ),
    ModalVolumeSpec(
        name="learn-nethack-hf-cache",
        mount_path=HF_CACHE_MOUNT_PATH,
        purpose="Hugging Face model, tokenizer, and dataset cache",
    ),
    ModalVolumeSpec(
        name="learn-nethack-watch",
        mount_path="/watch",
        purpose="read-only watcher state and rollout frame events",
    ),
)

SECRET_ENV_VARS = ("HF_TOKEN", "WANDB_API_KEY")
HF_TOKEN_ENV_ALIASES = (
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HF_HUB_TOKEN",
)

APT_PACKAGES = (
    "build-essential",
    "git",
    "cmake",
    "bison",
    "flex",
    "ffmpeg",
    "libbz2-dev",
    "liblzma-dev",
    "libncursesw5-dev",
    "pkg-config",
    "zlib1g-dev",
)

MODAL_TRAIN_PIP_PACKAGES = (
    "wandb>=0.17",
    "torch>=2.3",
    "transformers>=4.42",
    "datasets>=2.20",
    "accelerate>=0.31",
    "peft>=0.11",
    "trl>=0.9",
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git",
    "gymnasium>=0.29",
    "nle>=1.3.0",
    "fastapi>=0.110",
    "uvicorn>=0.30",
    "websockets>=12",
)

MODAL_VLLM_PIP_PACKAGES = ("vllm>=0.5",)


def run_artifact_layout(run_id: str) -> dict[str, str]:
    """Return canonical Modal run artifact paths for a run id."""
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a non-empty path segment")

    root = PurePosixPath("/runs") / run_id
    return {
        "root": str(root),
        "report": str(root / "reports" / "modal_readiness_report.json"),
        "wandb": str(root / "wandb"),
        "adapter": str(root / "adapters"),
        "ttyrec": str(root / "ttyrec"),
        "replay": str(root / "replay"),
        "watch": str(PurePosixPath("/watch") / run_id),
    }


def resolve_wandb_mode(env: Mapping[str, str | None]) -> WandbMode:
    """Resolve W&B mode and reject silent credentialless online runs."""
    requested_mode = (env.get("WANDB_MODE") or "online").lower()
    has_api_key = bool(env.get("WANDB_API_KEY"))

    if requested_mode == "offline":
        return WandbMode(mode="offline", requires_api_key=False)
    if not has_api_key:
        raise RuntimeError(
            "WANDB_API_KEY is required for online Modal runs; set "
            "WANDB_MODE=offline only for explicit smoke/test commands."
        )
    return WandbMode(mode="online", requires_api_key=True)


def modal_secret_names_for_env(env: Mapping[str, str | None]) -> tuple[str, ...]:
    """Return Modal secrets required for the requested readiness mode."""
    requested_mode = (env.get("WANDB_MODE") or "online").lower()
    if requested_mode == "offline":
        return ()
    return MODAL_SECRET_NAMES


def modal_function_env(env: Mapping[str, str | None]) -> dict[str, str]:
    """Return non-secret environment variables to inject into Modal functions."""
    requested_mode = (env.get("WANDB_MODE") or "online").lower()
    if requested_mode == "offline":
        return {"WANDB_MODE": "offline"}
    return {}


def modal_hf_cache_env() -> dict[str, str]:
    """Return Hugging Face cache variables backed by the Modal cache volume."""
    return {
        "HF_HOME": HF_CACHE_MOUNT_PATH,
        "HF_HUB_CACHE": f"{HF_CACHE_MOUNT_PATH}/hub",
        "HF_DATASETS_CACHE": f"{HF_CACHE_MOUNT_PATH}/datasets",
        "TRANSFORMERS_CACHE": HF_CACHE_MOUNT_PATH,
    }


def normalize_hf_token_env(env: MutableMapping[str, str]) -> bool:
    """Populate HF_TOKEN from common secret aliases when HF_TOKEN is absent."""
    if env.get("HF_TOKEN"):
        return False
    for alias in HF_TOKEN_ENV_ALIASES:
        token = env.get(alias)
        if token:
            env["HF_TOKEN"] = token
            return True
    return False


def modal_cloud_execution_context(
    *,
    function_call_id: str | None,
    input_id: str | None,
    env: Mapping[str, str | None],
    hostname: str,
) -> dict[str, str | None]:
    """Return proof that readiness executed inside a Modal cloud function."""
    if not function_call_id or not input_id:
        raise RuntimeError(
            "Modal readiness must run inside a Modal cloud function; use "
            "`modal run src/learn_nethack/modal_train.py::readiness ...`."
        )

    return {
        "backend": "modal_cloud",
        "function_call_id": function_call_id,
        "input_id": input_id,
        "task_id": env.get("MODAL_TASK_ID"),
        "hostname": hostname,
    }


def smoke_commands(run_id: str = "modal-readiness-smoke") -> dict[str, str]:
    """Return exact local commands for Modal readiness checks."""
    base = f"modal run src/learn_nethack/modal_train.py::readiness --run-id {run_id}"
    return {
        "online_readiness": base,
        "offline_readiness": f"WANDB_MODE=offline {base}",
        "unit_gate": "PYTHONPATH=src python3 -m unittest tests/test_modal_readiness.py -q",
        "pytest_gate": "uv run pytest tests/test_modal_readiness.py -q",
    }


def modal_volume_mounts() -> dict[str, str]:
    """Return mount path to Modal volume name mapping."""
    return {volume.mount_path: volume.name for volume in MODAL_VOLUMES}


def full_run_commands(
    *,
    run_id: str,
    local_dataset_root: str,
    action_manifest_path: str = "artifacts/action_manifest.json",
    dataset_remote_root: str = "/nld/",
    remote_db: str = "/datasets/nld/nld-aa-taster/ttyrecs.db",
    remote_nle_root: str = (
        "/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data"
    ),
    action_manifest_volume_path: str = "/action_manifest.json",
    remote_action_manifest: str = "/datasets/action_manifest.json",
) -> dict[str, str]:
    """Return exact commands for the full fine-tune proof sequence."""
    return {
        "upload_dataset": (
            "modal volume put learn-nethack-datasets "
            f"{local_dataset_root} {dataset_remote_root}"
        ),
        "upload_action_manifest": (
            "modal volume put --force learn-nethack-datasets "
            f"{action_manifest_path} {action_manifest_volume_path}"
        ),
        "eval_baseline": (
            "modal run src/learn_nethack/modal_train.py::sft_eval "
            f"--run-id {run_id}-baseline "
            f"--db {remote_db} "
            f"--action-manifest {remote_action_manifest} "
            f"--nle-root {remote_nle_root} "
            "--split validation "
            "--model-role baseline"
        ),
        "train": (
            "modal run src/learn_nethack/modal_train.py::sft_train "
            f"--run-id {run_id} "
            f"--db {remote_db} "
            f"--action-manifest {remote_action_manifest} "
            f"--nle-root {remote_nle_root} "
            "--mode single_frame "
            "--full-dataset"
        ),
        "eval_trained": (
            "modal run src/learn_nethack/modal_train.py::sft_eval "
            f"--run-id {run_id}-trained "
            f"--db {remote_db} "
            f"--action-manifest {remote_action_manifest} "
            f"--nle-root {remote_nle_root} "
            "--split validation "
            f"--adapter /runs/{run_id}/adapters "
            "--model-role trained"
        ),
        "compare": (
            "modal run src/learn_nethack/modal_train.py::sft_compare "
            f"--baseline /runs/{run_id}-baseline/reports/sft_eval_metrics.json "
            f"--trained /runs/{run_id}-trained/reports/sft_eval_metrics.json "
            f"--out /runs/{run_id}/reports/score_to_beat.json "
            f"--trained-run-id {run_id}-trained "
            f"--baseline-run-id {run_id}-baseline"
        ),
    }


def watch_compare_commands(
    *,
    run_id: str,
    action_manifest_path: str = "artifacts/action_manifest.json",
    action_manifest_volume_path: str = "/action_manifest.json",
    remote_action_manifest: str = "/datasets/action_manifest.json",
    env_id: str = DEFAULT_WATCH_ENV_ID,
    model_name: str = DEFAULT_WATCH_MODEL_NAME,
    max_steps: int = 10,
    current_checkpoint: str | None = None,
) -> dict[str, str]:
    """Return exact commands for a Modal side-by-side watch run."""
    current_checkpoint_arg = (
        f" --current-checkpoint {current_checkpoint}" if current_checkpoint else ""
    )
    return {
        "upload_action_manifest": (
            "modal volume put learn-nethack-datasets "
            f"{action_manifest_path} {action_manifest_volume_path}"
        ),
        "run_watch_compare": (
            "modal run src/learn_nethack/modal_train.py::watch_compare "
            f"--run-id {run_id} "
            f"--action-manifest {remote_action_manifest} "
            f"--env-id {env_id} "
            f"--model-name {model_name} "
            f"--max-steps {max_steps}"
            f"{current_checkpoint_arg}"
        ),
    }


def archive_full_run_commands(
    *,
    run_id: str,
    archive_manifest_path: str,
    action_manifest_path: str = "artifacts/action_manifest.json",
    archive_manifest_volume_path: str = "/nld-nao-archive.jsonl",
    remote_archive_manifest: str = "/datasets/nld-nao-archive.jsonl",
    action_manifest_volume_path: str = "/action_manifest.json",
    remote_action_manifest: str = "/datasets/action_manifest.json",
) -> dict[str, str]:
    """Return exact commands for archive-backed full fine-tune proof."""
    return {
        "upload_archive_manifest": (
            "modal volume put learn-nethack-datasets "
            f"{archive_manifest_path} {archive_manifest_volume_path}"
        ),
        "upload_action_manifest": (
            "modal volume put learn-nethack-datasets "
            f"{action_manifest_path} {action_manifest_volume_path}"
        ),
        "eval_baseline": (
            "modal run src/learn_nethack/modal_train.py::sft_eval "
            f"--run-id {run_id}-baseline "
            f"--action-manifest {remote_action_manifest} "
            f"--archive-manifest {remote_archive_manifest} "
            "--split validation "
            "--model-role baseline"
        ),
        "train": (
            "modal run src/learn_nethack/modal_train.py::sft_train "
            f"--run-id {run_id} "
            f"--action-manifest {remote_action_manifest} "
            f"--archive-manifest {remote_archive_manifest} "
            "--mode single_frame "
            "--full-dataset"
        ),
        "eval_trained": (
            "modal run src/learn_nethack/modal_train.py::sft_eval "
            f"--run-id {run_id}-trained "
            f"--action-manifest {remote_action_manifest} "
            f"--archive-manifest {remote_archive_manifest} "
            "--split validation "
            f"--adapter /runs/{run_id}/adapters "
            "--model-role trained"
        ),
        "compare": (
            "modal run src/learn_nethack/modal_train.py::sft_compare "
            f"--baseline /runs/{run_id}-baseline/reports/sft_eval_metrics.json "
            f"--trained /runs/{run_id}-trained/reports/sft_eval_metrics.json "
            f"--out /runs/{run_id}/reports/score_to_beat.json "
            f"--trained-run-id {run_id}-trained "
            f"--baseline-run-id {run_id}-baseline"
        ),
    }


def sft_existing_dataset_followup_commands(
    *,
    build_run_id: str,
    train_run_id: str,
    app_id: str | None = None,
    local_artifact_root: str = "artifacts",
    model_name: str = "google/gemma-4-E4b-it",
    max_steps: int = 0,
    training_objective: str = "policy_dynamics_phased",
    remote_action_manifest: str = "/datasets/action_manifest.json",
    remote_archive_manifest: str = "/datasets/nld-nao-archive.jsonl",
    baseline_eval_run_id: str | None = None,
    trained_eval_run_id: str | None = None,
    max_eval_rows: int = 512,
    eval_mode: str = "single_frame",
    eval_batch_size: int = 4,
    eval_seq_length: int = 64,
    eval_tasks: str = "policy_action,next_frame",
    label_source: str = "pseudo_visible_player_delta",
    next_frame_eval_mode: str = "both",
    next_frame_max_new_tokens: int = DEFAULT_NEXT_FRAME_MAX_NEW_TOKENS,
    next_frame_generate_max_rows: int = 64,
    next_frame_sequence_horizons: str = "1,5,10",
    next_frame_sequence_max_windows: int = 64,
    watch_run_id: str | None = None,
    watch_action_manifest: str = DEFAULT_WATCH_ACTION_MANIFEST,
    watch_env_id: str = DEFAULT_WATCH_ENV_ID,
    watch_character: str = "mon-hum-neu-mal",
    watch_seeds: str = DEFAULT_WATCH_PROOF_SEEDS,
    watch_max_steps: int = 80,
    runs_volume: str = "learn-nethack-runs",
    watch_volume: str = "learn-nethack-watch",
) -> dict[str, str]:
    """Return commands for pulling build reports and training built SFT JSONL."""
    remote_dataset_dir = f"/runs/{build_run_id}/sft-data"
    baseline_run = baseline_eval_run_id or f"{train_run_id}-baseline-eval"
    trained_run = trained_eval_run_id or f"{train_run_id}-trained-eval"
    watch_run = watch_run_id or f"{train_run_id}-watch-score-damage"
    local_root = local_artifact_root.rstrip("/")
    local_dir = f"{local_root}/{build_run_id}"
    local_train_dir = f"{local_root}/{train_run_id}"
    local_watch_dir = f"{local_root}/watch/{watch_run}"
    commands = {
        "pull_build_report": (
            f"modal volume get {runs_volume} "
            f"{build_run_id}/reports/sft_build_report.json "
            f"{local_dir}/sft_build_report.json"
        ),
        "pull_manifest": (
            f"modal volume get {runs_volume} "
            f"{build_run_id}/sft-data/manifest.json {local_dir}/manifest.json"
        ),
        "pull_rejection_report": (
            f"modal volume get {runs_volume} "
            f"{build_run_id}/sft-data/rejection_report.json "
            f"{local_dir}/rejection_report.json"
        ),
        "pull_sample_rows": (
            f"modal volume get {runs_volume} "
            f"{build_run_id}/sft-data/sample_rows.jsonl {local_dir}/sample_rows.jsonl"
        ),
        "train_existing": (
            "modal run src/learn_nethack/modal_train.py::sft_train_existing "
            f"--run-id {train_run_id} "
            f"--dataset-dir {remote_dataset_dir} "
            f"--model-name {model_name} "
            f"--max-steps {max_steps} "
            f"--training-objective {training_objective}"
        ),
        "eval_baseline_policy_and_next_frame": (
            "modal run src/learn_nethack/modal_train.py::sft_eval "
            f"--run-id {baseline_run} "
            f"--action-manifest {remote_action_manifest} "
            f"--dataset-dir {remote_dataset_dir} "
            "--split validation "
            "--model-role baseline "
            f"--model-name {model_name} "
            f"--mode {eval_mode} "
            f"--max-rows {max_eval_rows} "
            f"--batch-size {eval_batch_size} "
            f"--seq-length {eval_seq_length} "
            f"--eval-tasks {eval_tasks} "
            f"--label-source {label_source} "
            f"--next-frame-eval-mode {next_frame_eval_mode} "
            f"--next-frame-max-new-tokens {next_frame_max_new_tokens} "
            f"--next-frame-generate-max-rows {next_frame_generate_max_rows} "
            f"--next-frame-sequence-horizons {next_frame_sequence_horizons} "
            f"--next-frame-sequence-max-windows {next_frame_sequence_max_windows}"
        ),
        "eval_trained_policy_and_next_frame": (
            "modal run src/learn_nethack/modal_train.py::sft_eval "
            f"--run-id {trained_run} "
            f"--action-manifest {remote_action_manifest} "
            f"--dataset-dir {remote_dataset_dir} "
            "--split validation "
            "--model-role trained "
            f"--adapter /runs/{train_run_id}/adapters "
            f"--model-name {model_name} "
            f"--mode {eval_mode} "
            f"--max-rows {max_eval_rows} "
            f"--batch-size {eval_batch_size} "
            f"--seq-length {eval_seq_length} "
            f"--eval-tasks {eval_tasks} "
            f"--label-source {label_source} "
            f"--next-frame-eval-mode {next_frame_eval_mode} "
            f"--next-frame-max-new-tokens {next_frame_max_new_tokens} "
            f"--next-frame-generate-max-rows {next_frame_generate_max_rows} "
            f"--next-frame-sequence-horizons {next_frame_sequence_horizons} "
            f"--next-frame-sequence-max-windows {next_frame_sequence_max_windows}"
        ),
        "compare_policy_and_next_frame": (
            "modal run src/learn_nethack/modal_train.py::sft_compare "
            f"--baseline /runs/{baseline_run}/reports/sft_eval_metrics.json "
            f"--trained /runs/{trained_run}/reports/sft_eval_metrics.json "
            f"--out /runs/{train_run_id}/reports/score_to_beat_policy_and_next_frame.json "
            f"--trained-run-id {trained_run} "
            f"--baseline-run-id {baseline_run}"
        ),
        "watch_compare_score_damage": (
            "modal run src/learn_nethack/modal_train.py::watch_compare_sweep "
            f"--run-id {watch_run} "
            f"--action-manifest {watch_action_manifest} "
            f"--current-checkpoint /runs/{train_run_id}/adapters "
            f"--model-name {model_name} "
            f"--env-id {watch_env_id} "
            f"--character {watch_character} "
            f"--seeds {watch_seeds} "
            f"--max-steps {watch_max_steps}"
        ),
        "pull_score_to_beat_policy_and_next_frame": (
            f"modal volume get {runs_volume} "
            f"{train_run_id}/reports/score_to_beat_policy_and_next_frame.json "
            f"{local_train_dir}/score_to_beat_policy_and_next_frame.json"
        ),
        "pull_watch_report": (
            f"modal volume get {watch_volume} {watch_run}/sweep_report.json "
            f"{local_watch_dir}/sweep_report.json"
        ),
        "proof_gate_policy_next_frame_and_watch": (
            "uv run nethack-gemma sft proof-gate "
            f"--score-to-beat {local_train_dir}/score_to_beat_policy_and_next_frame.json "
            f"--watch-report {local_watch_dir}/sweep_report.json "
            f"--out {local_train_dir}/training_proof_gate.json"
        ),
    }
    if app_id:
        commands = {"logs": f"modal app logs {app_id}", **commands}
    return commands
