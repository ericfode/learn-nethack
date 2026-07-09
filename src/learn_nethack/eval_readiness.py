"""Readiness reports for Modal SFT evaluation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable, Mapping, Sequence


RUNS_VOLUME_NAME = "learn-nethack-runs"
EVAL_STATUS_SCHEMA_VERSION = "learn-nethack.sft-eval-status.v1"


@dataclass(frozen=True)
class EvalMarkerSpec:
    name: str
    run_relative_path: str
    local_filename: str
    required: bool


EVAL_MARKERS = (
    EvalMarkerSpec(
        name="contract",
        run_relative_path="reports/sft_eval_contract.json",
        local_filename="sft_eval_contract.json",
        required=True,
    ),
    EvalMarkerSpec(
        name="metrics",
        run_relative_path="reports/sft_eval_metrics.json",
        local_filename="sft_eval_metrics.json",
        required=True,
    ),
    EvalMarkerSpec(
        name="report",
        run_relative_path="reports/sft_eval_report.json",
        local_filename="sft_eval_report.json",
        required=True,
    ),
    EvalMarkerSpec(
        name="progress",
        run_relative_path="reports/sft_eval_progress.jsonl",
        local_filename="sft_eval_progress.jsonl",
        required=False,
    ),
)


Runner = Callable[[Sequence[str]], Mapping[str, Any]]


def build_eval_status_report(
    *,
    eval_run_id: str,
    local_artifact_root: Path,
    remote_marker_status: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a durable report describing whether an SFT eval is complete."""
    local_dir = local_artifact_root / eval_run_id
    markers: dict[str, Any] = {}
    missing_markers: list[str] = []
    for spec in EVAL_MARKERS:
        local_path = local_dir / spec.local_filename
        remote_status = dict((remote_marker_status or {}).get(spec.name) or {})
        status = _marker_status(local_path=local_path, remote_status=remote_status)
        if spec.required and status != "present":
            missing_markers.append(spec.name)
        markers[spec.name] = {
            "name": spec.name,
            "required": spec.required,
            "remote_path": f"/runs/{eval_run_id}/{spec.run_relative_path}",
            "volume_path": f"{eval_run_id}/{spec.run_relative_path}",
            "local_path": str(local_path),
            "local_exists": local_path.exists(),
            "status": status,
            "remote_status": remote_status or None,
            "modal_get_command": _modal_get_command(
                volume_path=f"{eval_run_id}/{spec.run_relative_path}",
                local_path=local_path,
            ),
        }
    return {
        "schema_version": EVAL_STATUS_SCHEMA_VERSION,
        "eval_run_id": eval_run_id,
        "remote_reports_dir": f"/runs/{eval_run_id}/reports",
        "local_dir": str(local_dir),
        "eval_ready": not missing_markers,
        "missing_markers": missing_markers,
        "markers": markers,
        "progress": summarize_eval_progress(local_dir / "sft_eval_progress.jsonl"),
        "next_action": (
            "run sft_compare or trained eval"
            if not missing_markers
            else "wait for or rerun SFT eval"
        ),
    }


def check_modal_eval_markers(
    *,
    eval_run_id: str,
    local_artifact_root: Path,
    runner: Runner | None = None,
) -> dict[str, dict[str, Any]]:
    """Pull small SFT eval artifacts from the Modal runs volume when present."""
    local_dir = local_artifact_root / eval_run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    effective_runner = runner or _run_subprocess
    statuses: dict[str, dict[str, Any]] = {}
    for spec in EVAL_MARKERS:
        command = [
            "modal",
            "volume",
            "get",
            "--force",
            RUNS_VOLUME_NAME,
            f"{eval_run_id}/{spec.run_relative_path}",
            str(local_dir / spec.local_filename),
        ]
        result = dict(effective_runner(command))
        statuses[spec.name] = {
            **result,
            "status": "present" if result.get("returncode") == 0 else "missing",
            "command": shlex.join(command),
        }
    return statuses


def summarize_eval_progress(progress_path: Path) -> dict[str, Any]:
    """Summarize generated-eval progress JSONL without requiring completion."""
    summary: dict[str, Any] = {
        "path": str(progress_path),
        "exists": progress_path.exists(),
        "event_count": 0,
        "latest": None,
        "max_evaluated_rows": 0,
        "max_generated_frames": 0,
        "max_parse_valid": 0,
    }
    if not progress_path.exists():
        return summary
    latest: dict[str, Any] | None = None
    event_count = 0
    max_evaluated_rows = 0
    max_generated_frames = 0
    max_parse_valid = 0
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
            max_evaluated_rows = max(
                max_evaluated_rows, _int_payload_value(payload, "evaluated_rows") or 0
            )
            max_generated_frames = max(
                max_generated_frames,
                _int_payload_value(payload, "generated_frames") or 0,
            )
            max_parse_valid = max(
                max_parse_valid, _int_payload_value(payload, "parse_valid") or 0
            )
        else:
            latest = {"parse_error": True, "raw": payload}
    summary["event_count"] = event_count
    summary["latest"] = latest
    summary["max_evaluated_rows"] = max_evaluated_rows
    summary["max_generated_frames"] = max_generated_frames
    summary["max_parse_valid"] = max_parse_valid
    return summary


def write_eval_status_report(report: Mapping[str, Any], out: Path) -> None:
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
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
