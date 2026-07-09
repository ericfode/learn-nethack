"""Readiness reports for full-dataset SFT build artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable, Mapping, Sequence


RUNS_VOLUME_NAME = "learn-nethack-runs"
FULL_BUILD_STATUS_SCHEMA_VERSION = "learn-nethack.sft-full-build-status.v1"


@dataclass(frozen=True)
class FullBuildMarkerSpec:
    name: str
    run_relative_path: str
    local_filename: str
    pull_small_file: bool


FULL_BUILD_MARKERS = (
    FullBuildMarkerSpec(
        name="train_jsonl",
        run_relative_path="sft-data/train.jsonl",
        local_filename="train.jsonl",
        pull_small_file=False,
    ),
    FullBuildMarkerSpec(
        name="manifest",
        run_relative_path="sft-data/manifest.json",
        local_filename="manifest.json",
        pull_small_file=True,
    ),
    FullBuildMarkerSpec(
        name="rejection_report",
        run_relative_path="sft-data/rejection_report.json",
        local_filename="rejection_report.json",
        pull_small_file=True,
    ),
    FullBuildMarkerSpec(
        name="sft_build_report",
        run_relative_path="reports/sft_build_report.json",
        local_filename="sft_build_report.json",
        pull_small_file=True,
    ),
)


Runner = Callable[[Sequence[str]], Mapping[str, Any]]


def build_full_build_readiness_report(
    *,
    build_run_id: str,
    local_artifact_root: Path,
    remote_marker_status: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a durable report describing whether a full build is train-ready."""
    local_dir = local_artifact_root / build_run_id
    markers: dict[str, Any] = {}
    missing_markers: list[str] = []
    for spec in FULL_BUILD_MARKERS:
        local_path = local_dir / spec.local_filename
        remote_status = dict((remote_marker_status or {}).get(spec.name) or {})
        status = _marker_status(local_path=local_path, remote_status=remote_status)
        if status != "present":
            missing_markers.append(spec.name)
        markers[spec.name] = {
            "name": spec.name,
            "required": True,
            "pull_small_file": spec.pull_small_file,
            "remote_path": f"/runs/{build_run_id}/{spec.run_relative_path}",
            "volume_path": f"{build_run_id}/{spec.run_relative_path}",
            "local_path": str(local_path),
            "local_exists": local_path.exists(),
            "status": status,
            "remote_status": remote_status or None,
            "modal_get_command": (
                _modal_get_command(
                    volume_path=f"{build_run_id}/{spec.run_relative_path}",
                    local_path=local_path,
                )
                if spec.pull_small_file
                else None
            ),
        }
    return {
        "schema_version": FULL_BUILD_STATUS_SCHEMA_VERSION,
        "build_run_id": build_run_id,
        "remote_dataset_dir": f"/runs/{build_run_id}/sft-data",
        "local_dir": str(local_dir),
        "train_ready": not missing_markers,
        "missing_markers": missing_markers,
        "markers": markers,
        "progress": summarize_full_build_progress(
            local_dir / "sft_build_progress.jsonl"
        ),
        "next_action": (
            "run sft_train_existing"
            if not missing_markers
            else "wait for or rerun full-dataset build before training"
        ),
    }


def check_modal_full_build_markers(
    *,
    build_run_id: str,
    local_artifact_root: Path,
    runner: Runner | None = None,
) -> dict[str, dict[str, Any]]:
    """Check required marker files in the Modal runs volume.

    Small JSON reports are pulled locally. The full train JSONL is checked by
    listing its parent directory instead of downloading the large file.
    """
    local_dir = local_artifact_root / build_run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    effective_runner = runner or _run_subprocess
    statuses: dict[str, dict[str, Any]] = {}
    for spec in FULL_BUILD_MARKERS:
        if spec.pull_small_file:
            command = [
                "modal",
                "volume",
                "get",
                "--force",
                RUNS_VOLUME_NAME,
                f"{build_run_id}/{spec.run_relative_path}",
                str(local_dir / spec.local_filename),
            ]
            result = dict(effective_runner(command))
            statuses[spec.name] = {
                **result,
                "status": "present" if result.get("returncode") == 0 else "missing",
                "command": shlex.join(command),
            }
            continue
        command = [
            "modal",
            "volume",
            "ls",
            RUNS_VOLUME_NAME,
            f"{build_run_id}/sft-data",
        ]
        result = dict(effective_runner(command))
        stdout = str(result.get("stdout") or "")
        present = result.get("returncode") == 0 and "train.jsonl" in stdout
        statuses[spec.name] = {
            **result,
            "status": "present" if present else "missing",
            "command": shlex.join(command),
        }
    progress_command = [
        "modal",
        "volume",
        "get",
        "--force",
        RUNS_VOLUME_NAME,
        f"{build_run_id}/reports/sft_build_progress.jsonl",
        str(local_dir / "sft_build_progress.jsonl"),
    ]
    progress_result = dict(effective_runner(progress_command))
    statuses["sft_build_progress"] = {
        **progress_result,
        "status": "present" if progress_result.get("returncode") == 0 else "missing",
        "command": shlex.join(progress_command),
    }
    return statuses


def summarize_full_build_progress(progress_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(progress_path),
        "exists": progress_path.exists(),
        "event_count": 0,
        "latest": None,
        "max_processed_transitions": 0,
        "max_accepted_policy_rows": 0,
        "max_accepted_next_frame_rows": 0,
        "max_rejected_rows": 0,
        "restart_count": 0,
    }
    if not progress_path.exists():
        return summary
    latest: dict[str, Any] | None = None
    event_count = 0
    previous_processed: int | None = None
    max_processed = 0
    max_accepted_policy = 0
    max_accepted_next_frame = 0
    max_rejected = 0
    restart_count = 0
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event_count += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {"parse_error": True, "raw": line}
        if isinstance(payload, dict):
            latest = payload
            processed = _int_payload_value(payload, "processed_transitions")
            accepted_policy = _int_payload_value(payload, "accepted_policy_rows")
            accepted_next_frame = _int_payload_value(
                payload, "accepted_next_frame_rows"
            )
            rejected = _int_payload_value(payload, "rejected_rows")
            if processed is not None:
                if previous_processed is not None and processed < previous_processed:
                    restart_count += 1
                previous_processed = processed
                max_processed = max(max_processed, processed)
            if accepted_policy is not None:
                max_accepted_policy = max(max_accepted_policy, accepted_policy)
            if accepted_next_frame is not None:
                max_accepted_next_frame = max(
                    max_accepted_next_frame, accepted_next_frame
                )
            if rejected is not None:
                max_rejected = max(max_rejected, rejected)
        else:
            latest = {"parse_error": True, "raw": payload}
    summary["event_count"] = event_count
    summary["latest"] = latest
    summary["max_processed_transitions"] = max_processed
    summary["max_accepted_policy_rows"] = max_accepted_policy
    summary["max_accepted_next_frame_rows"] = max_accepted_next_frame
    summary["max_rejected_rows"] = max_rejected
    summary["restart_count"] = restart_count
    return summary


def write_full_build_readiness_report(report: Mapping[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _marker_status(*, local_path: Path, remote_status: Mapping[str, Any]) -> str:
    status = remote_status.get("status")
    if status:
        return str(status)
    return "present" if local_path.exists() else "missing"


def _int_payload_value(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _modal_get_command(*, volume_path: str, local_path: Path) -> str:
    return shlex.join(
        [
            "modal",
            "volume",
            "get",
            "--force",
            RUNS_VOLUME_NAME,
            volume_path,
            str(local_path),
        ]
    )


def _run_subprocess(command: Sequence[str]) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
