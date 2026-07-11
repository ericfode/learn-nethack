"""Strict integrity audit for completed SFT JSONL datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from learn_nethack.action_manifest import load_action_manifest
from learn_nethack.sft_integrity_labels import (
    action_distribution,
    audit_next_frame_label,
    audit_policy_label,
    count_jsonl_rows,
    integer,
    messages_have_final_assistant,
    row_identity,
)


SPLITS = ("train", "validation", "test")
TASKS = ("policy_action", "next_frame")
SFT_INTEGRITY_SCHEMA_VERSION = "learn-nethack.sft-integrity.v2"


def audit_sft_dataset(
    dataset_dir: str | Path,
    *,
    expected_env_id: str | None = None,
) -> dict[str, Any]:
    """Audit split isolation and true-keypress label contracts for one dataset."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    action_manifest_path = root / "action_manifest.json"
    split_manifest_path = root / "split_manifest.json"
    manifest = _read_json_object(manifest_path)
    action_manifest = load_action_manifest(action_manifest_path)
    split_manifest = _read_json_object(split_manifest_path)
    failures: Counter[str] = Counter()
    failure_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def fail(reason: str, **example: Any) -> None:
        failures[reason] += 1
        if len(failure_examples[reason]) < 5:
            failure_examples[reason].append(example)

    manifest_env_id = manifest.get("env_id")
    if manifest_env_id != action_manifest.env_id:
        fail(
            "manifest_action_env_mismatch",
            manifest_env_id=manifest_env_id,
            action_manifest_env_id=action_manifest.env_id,
        )
    if expected_env_id is not None and action_manifest.env_id != expected_env_id:
        fail(
            "expected_env_mismatch",
            expected_env_id=expected_env_id,
            actual_env_id=action_manifest.env_id,
        )
    if manifest.get("split_row_limits") is not None and (
        manifest.get("split_limits_satisfied") is not True
    ):
        fail("split_limits_not_satisfied")

    valid_action_ids = action_manifest.valid_action_ids()
    valid_action_id_set = set(valid_action_ids)
    raw_key_to_action_ids: dict[int, list[int]] = defaultdict(list)
    action_names: dict[int, str] = {}
    for entry in action_manifest.entries:
        raw_key_to_action_ids[int(entry.raw_key_code)].append(int(entry.action_id))
        action_names[int(entry.action_id)] = entry.nle_action_name

    assigned_games = {
        split: {int(value) for value in _list_value(split_manifest, split, fail)}
        for split in SPLITS
    }
    _audit_cross_split_sets(
        assigned_games,
        reason="split_manifest_game_overlap",
        fail=fail,
    )

    actual_games = {split: set() for split in SPLITS}
    actual_episodes = {split: set() for split in SPLITS}
    task_row_counts = {split: {task: 0 for task in TASKS} for split in SPLITS}
    combined_row_counts: dict[str, int] = {}
    task_file_fingerprints: dict[str, dict[str, Any]] = {}
    action_histograms = {split: Counter() for split in SPLITS}
    role_histograms = {split: Counter() for split in SPLITS}
    seen_row_ids: dict[str, set[tuple[Any, ...]]] = {task: set() for task in TASKS}
    transition_actions: dict[str, dict[tuple[Any, ...], int]] = {
        task: {} for task in TASKS
    }
    dynamics_changed_target_count = 0
    dynamics_comparable_target_count = 0

    for split in SPLITS:
        for task in TASKS:
            path = root / f"{split}.{task}.jsonl"
            fingerprint = _audit_task_file(
                path=path,
                split=split,
                task=task,
                assigned_games=assigned_games,
                actual_games=actual_games,
                actual_episodes=actual_episodes,
                task_row_counts=task_row_counts,
                action_histograms=action_histograms,
                role_histograms=role_histograms,
                seen_row_ids=seen_row_ids,
                transition_actions=transition_actions,
                valid_action_ids=valid_action_ids,
                valid_action_id_set=valid_action_id_set,
                raw_key_to_action_ids=raw_key_to_action_ids,
                fail=fail,
            )
            task_file_fingerprints[path.name] = fingerprint
            dynamics_changed_target_count += int(
                fingerprint.pop("dynamics_changed_target_count", 0)
            )
            dynamics_comparable_target_count += int(
                fingerprint.pop("dynamics_comparable_target_count", 0)
            )

        combined_path = root / f"{split}.jsonl"
        combined_count = count_jsonl_rows(combined_path, fail=fail)
        combined_row_counts[split] = combined_count
        expected_combined = sum(task_row_counts[split].values())
        if combined_count != expected_combined:
            fail(
                "combined_row_count_mismatch",
                split=split,
                expected=expected_combined,
                actual=combined_count,
            )

    expected_counts = manifest.get("accepted_rows_by_split")
    if not isinstance(expected_counts, Mapping):
        fail("manifest_split_counts_missing")
    else:
        for split in SPLITS:
            split_counts = expected_counts.get(split)
            for task in TASKS:
                expected = (
                    split_counts.get(task)
                    if isinstance(split_counts, Mapping)
                    else None
                )
                actual = task_row_counts[split][task]
                if expected != actual:
                    fail(
                        "task_row_count_mismatch",
                        split=split,
                        task=task,
                        expected=expected,
                        actual=actual,
                    )

    _audit_cross_split_sets(
        actual_games,
        reason="actual_game_overlap",
        fail=fail,
    )
    _audit_cross_split_sets(
        actual_episodes,
        reason="actual_episode_overlap",
        fail=fail,
    )

    paired_keys = set(transition_actions["policy_action"]) & set(
        transition_actions["next_frame"]
    )
    paired_action_mismatch_count = 0
    for row_id in paired_keys:
        policy_action = transition_actions["policy_action"][row_id]
        dynamics_action = transition_actions["next_frame"][row_id]
        if policy_action != dynamics_action:
            paired_action_mismatch_count += 1
            fail(
                "paired_transition_action_mismatch",
                row_id=list(row_id),
                policy_action_id=policy_action,
                dynamics_action_id=dynamics_action,
            )

    roles_by_split = {
        split: {role for role in histogram if role != "<missing>"}
        for split, histogram in role_histograms.items()
    }
    required_roles = roles_by_split["train"]
    missing_roles_by_split: dict[str, list[str]] = {}
    for split in ("validation", "test"):
        missing = sorted(required_roles - roles_by_split[split])
        missing_roles_by_split[split] = missing
        if missing:
            fail(
                "heldout_role_coverage_missing",
                split=split,
                required_roles=sorted(required_roles),
                missing_roles=missing,
            )

    total_policy_rows = sum(task_row_counts[split]["policy_action"] for split in SPLITS)
    total_dynamics_rows = sum(task_row_counts[split]["next_frame"] for split in SPLITS)
    aggregate_actions = sum(action_histograms.values(), Counter())
    report = {
        "schema_version": SFT_INTEGRITY_SCHEMA_VERSION,
        "dataset_dir": str(root),
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "expected_env_id": expected_env_id,
        "env_id": action_manifest.env_id,
        "label_source": "true_nld_keypress_and_successor_frame",
        "counts": {
            "policy_rows": total_policy_rows,
            "next_frame_rows": total_dynamics_rows,
            "paired_transition_rows": len(paired_keys),
            "policy_only_transition_rows": len(
                set(transition_actions["policy_action"]) - paired_keys
            ),
            "next_frame_only_transition_rows": len(
                set(transition_actions["next_frame"]) - paired_keys
            ),
            "paired_action_mismatches": paired_action_mismatch_count,
            "failure_count": sum(failures.values()),
        },
        "row_counts_by_split": task_row_counts,
        "combined_row_counts_by_split": combined_row_counts,
        "actual_game_counts_by_split": {
            split: len(values) for split, values in actual_games.items()
        },
        "actual_episode_counts_by_split": {
            split: len(values) for split, values in actual_episodes.items()
        },
        "action_distribution": action_distribution(
            aggregate_actions,
            action_names=action_names,
        ),
        "action_distribution_by_split": {
            split: action_distribution(histogram, action_names=action_names)
            for split, histogram in action_histograms.items()
        },
        "role_distribution_by_split": {
            split: dict(sorted(histogram.items()))
            for split, histogram in role_histograms.items()
        },
        "role_coverage": {
            "required_roles": sorted(required_roles),
            "roles_by_split": {
                split: sorted(roles) for split, roles in roles_by_split.items()
            },
            "missing_roles_by_split": missing_roles_by_split,
        },
        "dynamics_diagnostics": {
            "current_target_comparable_rows": dynamics_comparable_target_count,
            "changed_target_rows": dynamics_changed_target_count,
            "changed_target_rate": _rate(
                dynamics_changed_target_count,
                dynamics_comparable_target_count,
            ),
        },
        "file_fingerprints": task_file_fingerprints,
        "failure_reasons": dict(sorted(failures.items())),
        "failure_examples": {
            reason: examples for reason, examples in sorted(failure_examples.items())
        },
    }
    return report


def write_sft_integrity_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _audit_task_file(
    *,
    path: Path,
    split: str,
    task: str,
    assigned_games: Mapping[str, set[int]],
    actual_games: dict[str, set[int]],
    actual_episodes: dict[str, set[str]],
    task_row_counts: dict[str, dict[str, int]],
    action_histograms: dict[str, Counter[int]],
    role_histograms: dict[str, Counter[str]],
    seen_row_ids: dict[str, set[tuple[Any, ...]]],
    transition_actions: dict[str, dict[tuple[Any, ...], int]],
    valid_action_ids: list[int],
    valid_action_id_set: set[int],
    raw_key_to_action_ids: Mapping[int, list[int]],
    fail: Any,
) -> dict[str, Any]:
    if not path.exists():
        fail("task_file_missing", path=str(path))
        return {"sha256": None, "size_bytes": 0, "row_count": 0}
    digest = hashlib.sha256()
    changed_target_count = 0
    comparable_target_count = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            task_row_counts[split][task] += 1
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                fail(
                    "malformed_json_row",
                    path=str(path),
                    line_number=line_number,
                    error=str(exc),
                )
                continue
            if not isinstance(row, Mapping):
                fail(
                    "row_not_object",
                    path=str(path),
                    line_number=line_number,
                )
                continue
            if row.get("split") != split:
                fail(
                    "row_split_mismatch",
                    path=str(path),
                    line_number=line_number,
                    row_split=row.get("split"),
                )
            if row.get("task") != task:
                fail(
                    "row_task_mismatch",
                    path=str(path),
                    line_number=line_number,
                    row_task=row.get("task"),
                )
            gameid = integer(row.get("gameid"))
            episode_id = row.get("episode_id")
            if gameid is None:
                fail("row_gameid_invalid", path=str(path), line_number=line_number)
            else:
                actual_games[split].add(gameid)
                if gameid not in assigned_games[split]:
                    fail(
                        "row_game_missing_from_split_manifest",
                        path=str(path),
                        line_number=line_number,
                        gameid=gameid,
                        split=split,
                    )
            if not isinstance(episode_id, str) or not episode_id:
                fail("row_episode_id_invalid", path=str(path), line_number=line_number)
            else:
                actual_episodes[split].add(episode_id)
            row_id = row_identity(row)
            if row_id in seen_row_ids[task]:
                fail(
                    "duplicate_task_row_identity",
                    path=str(path),
                    line_number=line_number,
                    row_id=list(row_id),
                )
            seen_row_ids[task].add(row_id)

            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                fail("row_metadata_invalid", path=str(path), line_number=line_number)
                continue
            if metadata.get("valid_action_ids") != valid_action_ids:
                fail(
                    "row_valid_action_ids_mismatch",
                    path=str(path),
                    line_number=line_number,
                )
            raw_key_code = integer(metadata.get("raw_key_code"))
            mapped_action_id: int | None = None
            if raw_key_code is None:
                fail(
                    "row_raw_key_code_missing",
                    path=str(path),
                    line_number=line_number,
                )
            else:
                candidates = raw_key_to_action_ids.get(raw_key_code, [])
                if not candidates:
                    fail(
                        "row_raw_key_unmapped",
                        path=str(path),
                        line_number=line_number,
                        raw_key_code=raw_key_code,
                    )
                elif len(candidates) != 1:
                    fail(
                        "row_raw_key_ambiguous",
                        path=str(path),
                        line_number=line_number,
                        raw_key_code=raw_key_code,
                        candidate_action_ids=candidates,
                    )
                else:
                    mapped_action_id = candidates[0]
            messages = row.get("messages")
            if not messages_have_final_assistant(messages):
                fail(
                    "row_messages_invalid",
                    path=str(path),
                    line_number=line_number,
                )
                continue
            role = metadata.get("role")
            role_histograms[split][str(role) if role else "<missing>"] += 1
            if task == "policy_action":
                target_action_id = audit_policy_label(
                    messages=messages,
                    metadata=metadata,
                    mapped_action_id=mapped_action_id,
                    valid_action_id_set=valid_action_id_set,
                    path=path,
                    line_number=line_number,
                    fail=fail,
                )
            else:
                target_action_id, changed = audit_next_frame_label(
                    messages=messages,
                    metadata=metadata,
                    mapped_action_id=mapped_action_id,
                    valid_action_id_set=valid_action_id_set,
                    path=path,
                    line_number=line_number,
                    fail=fail,
                )
                if changed is not None:
                    comparable_target_count += 1
                    changed_target_count += int(changed)
            if target_action_id is not None:
                transition_actions[task][row_id] = target_action_id
                if task == "policy_action":
                    action_histograms[split][target_action_id] += 1

    return {
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "row_count": task_row_counts[split][task],
        "dynamics_changed_target_count": changed_target_count,
        "dynamics_comparable_target_count": comparable_target_count,
    }


def _audit_cross_split_sets(
    values_by_split: Mapping[str, set[Any]],
    *,
    reason: str,
    fail: Any,
) -> None:
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = values_by_split[left] & values_by_split[right]
            if overlap:
                fail(
                    reason,
                    left_split=left,
                    right_split=right,
                    overlap_count=len(overlap),
                    overlap_sample=sorted(overlap, key=str)[:10],
                )


def _list_value(payload: Mapping[str, Any], key: str, fail: Any) -> list[Any]:
    value = payload.get(key)
    if isinstance(value, list):
        return value
    fail("split_manifest_list_missing", split=key)
    return []


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required SFT integrity input is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
