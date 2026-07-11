"""Small deterministic SFT fixtures for real-tokenizer overfit gates."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any


def build_sft_overfit_fixture(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    rows_per_task: int = 4,
) -> dict[str, Any]:
    """Build an exact policy/dynamics subset from an existing safe SFT artifact."""
    if rows_per_task <= 0:
        raise ValueError("rows_per_task must be positive")
    source = Path(source_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected_by_task = {
        task: _read_task_rows(
            source / f"train.{task}.jsonl",
            task=task,
            limit=rows_per_task,
        )
        for task in ("policy_action", "next_frame")
    }
    for task, rows in selected_by_task.items():
        _write_rows(target / f"train.{task}.jsonl", rows)
    combined = [
        row
        for index in range(rows_per_task)
        for row in (
            selected_by_task["policy_action"][index],
            selected_by_task["next_frame"][index],
        )
    ]
    _write_rows(target / "train.jsonl", combined)

    source_manifest_path = source / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    manifest = {
        "schema_version": "learn-nethack.sft-overfit-fixture.v1",
        "source_dir": str(source),
        "source_dataset_name": source_manifest.get("dataset_name"),
        "label_source": source_manifest.get("label_source", "true_keypress"),
        "tasks": ["policy_action", "next_frame"],
        "accepted_policy_rows": rows_per_task,
        "accepted_next_frame_rows": rows_per_task,
        "rejected_rows": 0,
        "combined_train_rows": len(combined),
    }
    _write_json(target / "manifest.json", manifest)
    _write_json(
        target / "rejection_report.json",
        {
            "schema_version": "learn-nethack.sft-rejections.v1",
            "total_rejected": 0,
            "reasons": {},
        },
    )
    action_manifest = source / "action_manifest.json"
    if action_manifest.exists():
        shutil.copyfile(action_manifest, target / "action_manifest.json")
    return {
        "schema_version": "learn-nethack.sft-overfit-build-report.v1",
        "output_dir": str(target),
        "manifest": manifest,
        "files": {
            "train": str(target / "train.jsonl"),
            "policy_action": str(target / "train.policy_action.jsonl"),
            "next_frame": str(target / "train.next_frame.jsonl"),
        },
    }


def _read_task_rows(path: Path, *, task: str, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required source task file is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != task:
                raise ValueError(f"row in {path} does not have task {task!r}")
            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError(f"row in {path} lacks a final assistant target")
            rows.append(row)
            if len(rows) >= limit:
                break
    if len(rows) != limit:
        raise ValueError(f"{path} contains only {len(rows)} usable rows; need {limit}")
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
