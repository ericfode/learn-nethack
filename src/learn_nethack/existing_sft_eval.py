"""Fail-closed loading of prebuilt SFT validation and test rows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from learn_nethack.action_manifest import ActionManifest, load_action_manifest
from learn_nethack.sft_integrity import audit_sft_dataset
from learn_nethack.sft_train import load_jsonl_rows


@dataclass(frozen=True)
class ExistingSftEvalDataset:
    dataset_dir: Path
    dataset_name: str
    mode: str
    split: str
    gameids: tuple[int, ...]
    policy_rows_path: Path
    next_frame_rows_path: Path
    policy_rows: list[dict[str, Any]]
    next_frame_rows: list[dict[str, Any]]
    manifest: dict[str, Any]
    integrity_report: dict[str, Any]


def load_existing_sft_eval_dataset(
    *,
    dataset_dir: str | Path,
    split: str,
    tasks: Sequence[str],
    mode: str,
    action_manifest: ActionManifest,
) -> ExistingSftEvalDataset:
    """Re-audit and load the exact named split from a prebuilt SFT dataset."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"existing SFT dataset directory not found: {root}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown eval split: {split}")

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"existing SFT manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("env_id") != action_manifest.env_id:
        raise ValueError(
            "existing SFT dataset environment mismatch: "
            f"dataset={manifest.get('env_id')!r}, "
            f"action_manifest={action_manifest.env_id!r}"
        )
    if manifest.get("mode") != mode:
        raise ValueError(
            "existing SFT dataset mode mismatch: "
            f"dataset={manifest.get('mode')!r}, requested={mode!r}"
        )

    embedded_action_manifest_path = root / "action_manifest.json"
    if not embedded_action_manifest_path.is_file():
        raise FileNotFoundError(
            f"existing SFT action manifest not found: {embedded_action_manifest_path}"
        )
    embedded_action_manifest = load_action_manifest(embedded_action_manifest_path)
    if embedded_action_manifest != action_manifest:
        raise ValueError(
            "existing SFT action manifest does not match the evaluator manifest"
        )

    integrity_report = audit_sft_dataset(
        root,
        expected_env_id=action_manifest.env_id,
    )
    if integrity_report.get("status") != "passed":
        raise RuntimeError(
            "existing SFT dataset failed integrity audit: "
            f"{integrity_report.get('failure_reasons')}"
        )

    policy_rows_path = root / f"{split}.policy_action.jsonl"
    next_frame_rows_path = root / f"{split}.next_frame.jsonl"
    task_set = set(tasks)
    policy_rows = _load_required_task_rows(
        policy_rows_path,
        task="policy_action",
        split=split,
        required="policy_action" in task_set,
    )
    next_frame_rows = _load_required_task_rows(
        next_frame_rows_path,
        task="next_frame",
        split=split,
        required="next_frame" in task_set,
    )
    all_rows = [*policy_rows, *next_frame_rows]
    gameids = tuple(sorted({int(row["gameid"]) for row in all_rows}))
    if not gameids:
        raise RuntimeError(f"existing SFT eval split {split!r} has no gameids")

    return ExistingSftEvalDataset(
        dataset_dir=root,
        dataset_name=str(manifest.get("dataset_name") or root.name),
        mode=mode,
        split=split,
        gameids=gameids,
        policy_rows_path=policy_rows_path,
        next_frame_rows_path=next_frame_rows_path,
        policy_rows=policy_rows,
        next_frame_rows=next_frame_rows,
        manifest=manifest,
        integrity_report=integrity_report,
    )


def _load_required_task_rows(
    path: Path,
    *,
    task: str,
    split: str,
    required: bool,
) -> list[dict[str, Any]]:
    if not required:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"existing SFT {task} rows not found: {path}")
    rows = load_jsonl_rows([path])
    if not rows:
        raise RuntimeError(f"existing SFT {task} rows are empty: {path}")
    for row_index, row in enumerate(rows):
        if row.get("task") != task or row.get("split") != split:
            raise ValueError(
                f"{path} row {row_index} has task/split "
                f"{row.get('task')!r}/{row.get('split')!r}, expected "
                f"{task!r}/{split!r}"
            )
    return rows
