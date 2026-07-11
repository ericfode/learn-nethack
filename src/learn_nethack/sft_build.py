"""Dataset writer for multi-task NLD SFT rows."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from learn_nethack.observations import render_observation_text
from learn_nethack.nld_decode import normalize_decoded_batch, normalize_frame_only_batch
from learn_nethack.nld_metadata import split_gameids
from learn_nethack.sft_rows import (
    HistoryBuffer,
    build_next_frame_row,
    build_pseudo_next_frame_row,
    build_pseudo_policy_action_row,
    build_policy_action_row,
    policy_feedback_from_outcome_observation,
)
from learn_nethack.pseudo_labels import infer_visible_movement_pseudo_label


@dataclass(frozen=True)
class SftBuildResult:
    accepted_policy_rows: int
    accepted_next_frame_rows: int
    rejected_rows: int
    output_dir: str


@dataclass(frozen=True)
class SftBuildProgress:
    processed_transitions: int
    accepted_policy_rows: int
    accepted_next_frame_rows: int
    rejected_rows: int
    reason: str
    last_gameid: int | None
    last_step: int | None


ProgressCallback = Callable[[SftBuildProgress], None]


@dataclass(frozen=True)
class SplitRowLimits:
    train: int
    validation: int
    test: int

    def as_dict(self) -> dict[str, int]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _split_for_gameid(gameid: int, splits: dict[str, set[int]]) -> str | None:
    for split_name, gameids in splits.items():
        if int(gameid) in gameids:
            return split_name
    return None


def write_sft_dataset(
    *,
    dataset_name: str,
    mode: str,
    transitions: Iterable,
    action_manifest,
    game_metadata_by_id: dict[int, dict],
    splits: dict[str, set[int]],
    out_dir: str | Path,
    max_rows: int | None = None,
    tasks: tuple[str, ...] = ("policy_action", "next_frame"),
    token_budget: int = 2048,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1000,
    split_row_limits: SplitRowLimits | None = None,
    game_order_strategy: str = "source_order",
) -> SftBuildResult:
    _validate_row_limit_contract(max_rows=max_rows, split_row_limits=split_row_limits)
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    rejection_reasons: Counter[str] = Counter()
    history = HistoryBuffer(max_items=16)
    accepted_policy_rows = 0
    accepted_next_frame_rows = 0
    processed_transitions = 0
    sample_rows: list[dict] = []
    accepted_rows_by_split = _empty_split_task_counts()
    split_quota_skips: Counter[str] = Counter()

    with ExitStack() as stack:
        handles: dict[tuple[str, str | None], Any] = {}
        for split_name in ("train", "validation", "test"):
            handles[(split_name, None)] = stack.enter_context(
                (target / f"{split_name}.jsonl").open("w", encoding="utf-8")
            )
            for task in ("policy_action", "next_frame"):
                handles[(split_name, task)] = stack.enter_context(
                    (target / f"{split_name}.{task}.jsonl").open(
                        "w",
                        encoding="utf-8",
                    )
                )

        for transition in transitions:
            processed_transitions += 1
            if split_row_limits is not None and _all_split_limits_reached(
                tasks=tasks,
                counts=accepted_rows_by_split,
                limits=split_row_limits,
            ):
                break
            if (
                split_row_limits is None
                and max_rows is not None
                and _accepted_rows_for_limit(
                    tasks=tasks,
                    accepted_policy_rows=accepted_policy_rows,
                    accepted_next_frame_rows=accepted_next_frame_rows,
                )
                >= max_rows
            ):
                break

            split = _split_for_gameid(transition.gameid, splits)
            if split is None:
                rejection_reasons["gameid_not_in_split"] += 1
                _maybe_report_progress(
                    progress_callback=progress_callback,
                    progress_interval=progress_interval,
                    processed_transitions=processed_transitions,
                    accepted_policy_rows=accepted_policy_rows,
                    accepted_next_frame_rows=accepted_next_frame_rows,
                    rejected_rows=sum(rejection_reasons.values()),
                    reason="gameid_not_in_split",
                    last_gameid=transition.gameid,
                    last_step=transition.step,
                )
                continue

            write_policy = "policy_action" in tasks and _split_task_has_capacity(
                split=split,
                task="policy_action",
                counts=accepted_rows_by_split,
                limits=split_row_limits,
            )
            write_next_frame = "next_frame" in tasks and _split_task_has_capacity(
                split=split,
                task="next_frame",
                counts=accepted_rows_by_split,
                limits=split_row_limits,
            )
            if not write_policy and not write_next_frame:
                split_quota_skips[split] += 1
                continue

            game_metadata = game_metadata_by_id.get(transition.gameid, {})
            current_history = history.history_for(
                gameid=transition.gameid,
                mode=mode,
                token_budget=token_budget,
            )

            action_id: int | None = None
            if write_policy or write_next_frame:
                try:
                    action_id = action_manifest.action_id_for_raw_key(
                        transition.raw_key_code
                    )
                except KeyError:
                    rejection_reasons["unmapped_raw_key_code"] += 1
                    _maybe_report_progress(
                        progress_callback=progress_callback,
                        progress_interval=progress_interval,
                        processed_transitions=processed_transitions,
                        accepted_policy_rows=accepted_policy_rows,
                        accepted_next_frame_rows=accepted_next_frame_rows,
                        rejected_rows=sum(rejection_reasons.values()),
                        reason="unmapped_raw_key_code",
                        last_gameid=transition.gameid,
                        last_step=transition.step,
                    )
                    continue
                except ValueError:
                    rejection_reasons["ambiguous_raw_key_code"] += 1
                    _maybe_report_progress(
                        progress_callback=progress_callback,
                        progress_interval=progress_interval,
                        processed_transitions=processed_transitions,
                        accepted_policy_rows=accepted_policy_rows,
                        accepted_next_frame_rows=accepted_next_frame_rows,
                        rejected_rows=sum(rejection_reasons.values()),
                        reason="ambiguous_raw_key_code",
                        last_gameid=transition.gameid,
                        last_step=transition.step,
                    )
                    continue
            if write_policy:
                policy_row = build_policy_action_row(
                    dataset_name=dataset_name,
                    split=split,
                    mode=mode,
                    transition=transition,
                    action_manifest=action_manifest,
                    game_metadata=game_metadata,
                    history=current_history,
                )
                _write_row(handles, policy_row)
                if split == "train" and len(sample_rows) < 10:
                    sample_rows.append(policy_row)
                accepted_policy_rows += 1
                accepted_rows_by_split[split]["policy_action"] += 1
                action_id = int(policy_row["metadata"]["target_action_id"])

            if write_next_frame:
                try:
                    next_frame_row = build_next_frame_row(
                        dataset_name=dataset_name,
                        split=split,
                        mode=mode,
                        transition=transition,
                        action_manifest=action_manifest,
                        game_metadata=game_metadata,
                        history=current_history,
                    )
                except ValueError as exc:
                    rejection_reasons[str(exc)] += 1
                else:
                    _write_row(handles, next_frame_row)
                    if split == "train" and len(sample_rows) < 10:
                        sample_rows.append(next_frame_row)
                    accepted_next_frame_rows += 1
                    accepted_rows_by_split[split]["next_frame"] += 1

            if action_id is not None:
                history.append(
                    gameid=transition.gameid,
                    observation_text=render_observation_text(transition.observation),
                    action_id=action_id,
                    feedback=policy_feedback_from_outcome_observation(
                        action_id=action_id,
                        observation=transition.next_observation,
                    ),
                )
            _maybe_report_progress(
                progress_callback=progress_callback,
                progress_interval=progress_interval,
                processed_transitions=processed_transitions,
                accepted_policy_rows=accepted_policy_rows,
                accepted_next_frame_rows=accepted_next_frame_rows,
                rejected_rows=sum(rejection_reasons.values()),
                reason="accepted",
                last_gameid=transition.gameid,
                last_step=transition.step,
            )

    if "policy_action" in tasks and accepted_policy_rows == 0:
        raise RuntimeError("SFT dataset build produced zero policy_action rows")
    if (
        "next_frame" in tasks
        and "policy_action" not in tasks
        and accepted_next_frame_rows == 0
    ):
        raise RuntimeError("SFT dataset build produced zero next_frame rows")

    total_rejected = sum(rejection_reasons.values())
    split_limits_satisfied = _split_limits_satisfied(
        tasks=tasks,
        counts=accepted_rows_by_split,
        limits=split_row_limits,
    )
    manifest = {
        "schema_version": "learn-nethack.sft-manifest.v1",
        "dataset_name": dataset_name,
        "mode": mode,
        "tasks": list(tasks),
        "env_id": action_manifest.env_id,
        "accepted_policy_rows": accepted_policy_rows,
        "accepted_next_frame_rows": accepted_next_frame_rows,
        "accepted_rows_by_split": accepted_rows_by_split,
        "split_row_limits": (
            split_row_limits.as_dict() if split_row_limits is not None else None
        ),
        "split_limits_satisfied": split_limits_satisfied,
        "split_quota_skips": dict(sorted(split_quota_skips.items())),
        "game_order_strategy": game_order_strategy,
        "token_budget": token_budget,
        "rejected_rows": total_rejected,
        "next_frame_status": "ok" if accepted_next_frame_rows else "no_rows",
    }
    _write_json(target / "manifest.json", manifest)
    _write_json(
        target / "rejection_report.json",
        {
            "schema_version": "learn-nethack.sft-rejections.v1",
            "total_rejected": total_rejected,
            "reasons": dict(sorted(rejection_reasons.items())),
        },
    )
    action_manifest.save(target / "action_manifest.json")
    _write_json(
        target / "split_manifest.json",
        {split: sorted(gameids) for split, gameids in splits.items()},
    )
    with (target / "sample_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _raise_for_unsatisfied_split_limits(
        tasks=tasks,
        counts=accepted_rows_by_split,
        limits=split_row_limits,
    )

    return SftBuildResult(
        accepted_policy_rows=accepted_policy_rows,
        accepted_next_frame_rows=accepted_next_frame_rows,
        rejected_rows=total_rejected,
        output_dir=str(target),
    )


def _write_row(handles: dict[tuple[str, str | None], Any], row: dict) -> None:
    split = row["split"]
    task = row["task"]
    line = json.dumps(row, sort_keys=True) + "\n"
    handles[(split, None)].write(line)
    handles[(split, task)].write(line)


def merge_sft_dataset_shards(
    *,
    shard_dirs: Sequence[str | Path],
    out_dir: str | Path,
) -> SftBuildResult:
    """Merge completed SFT shard directories into a standard trainable dataset."""
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    shard_paths = [Path(path) for path in shard_dirs]
    manifests = [_read_required_json(path / "manifest.json") for path in shard_paths]
    rejection_reports = [
        _read_required_json(path / "rejection_report.json") for path in shard_paths
    ]
    first_manifest = manifests[0]
    tasks = tuple(str(task) for task in first_manifest.get("tasks", []))
    if not tasks:
        raise ValueError("shard manifest tasks must be non-empty")
    for shard_path, manifest in zip(shard_paths, manifests, strict=True):
        _validate_merge_compatible_manifest(
            shard_path=shard_path,
            manifest=manifest,
            first_manifest=first_manifest,
        )

    for split_name in ("train", "validation", "test"):
        _concat_jsonl_files(
            [path / f"{split_name}.jsonl" for path in shard_paths],
            target / f"{split_name}.jsonl",
        )
        for task in ("policy_action", "next_frame"):
            _concat_jsonl_files(
                [path / f"{split_name}.{task}.jsonl" for path in shard_paths],
                target / f"{split_name}.{task}.jsonl",
            )

    accepted_policy_rows = sum(
        int(manifest.get("accepted_policy_rows", 0) or 0) for manifest in manifests
    )
    accepted_next_frame_rows = sum(
        int(manifest.get("accepted_next_frame_rows", 0) or 0) for manifest in manifests
    )
    rejected_rows = sum(
        int(manifest.get("rejected_rows", 0) or 0) for manifest in manifests
    )
    rejection_reasons: Counter[str] = Counter()
    for report in rejection_reports:
        reasons = report.get("reasons", {})
        if isinstance(reasons, dict):
            for reason, count in reasons.items():
                rejection_reasons[str(reason)] += int(count or 0)

    merged_manifest = {
        **{
            key: value
            for key, value in first_manifest.items()
            if key
            not in {
                "accepted_policy_rows",
                "accepted_next_frame_rows",
                "rejected_rows",
                "next_frame_status",
            }
        },
        "schema_version": "learn-nethack.sft-manifest.v1",
        "accepted_policy_rows": accepted_policy_rows,
        "accepted_next_frame_rows": accepted_next_frame_rows,
        "rejected_rows": rejected_rows,
        "next_frame_status": "ok" if accepted_next_frame_rows else "no_rows",
        "shard_count": len(shard_paths),
        "shards": [str(path) for path in shard_paths],
    }
    _write_json(target / "manifest.json", merged_manifest)
    _write_json(
        target / "rejection_report.json",
        {
            "schema_version": "learn-nethack.sft-rejections.v1",
            "total_rejected": rejected_rows,
            "reasons": dict(sorted(rejection_reasons.items())),
            "shard_count": len(shard_paths),
        },
    )
    _merge_split_manifests(shard_paths=shard_paths, target=target)
    _concat_jsonl_files(
        [path / "sample_rows.jsonl" for path in shard_paths],
        target / "sample_rows.jsonl",
        max_lines=10,
    )
    action_manifest = shard_paths[0] / "action_manifest.json"
    if action_manifest.exists():
        (target / "action_manifest.json").write_text(
            action_manifest.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return SftBuildResult(
        accepted_policy_rows=accepted_policy_rows,
        accepted_next_frame_rows=accepted_next_frame_rows,
        rejected_rows=rejected_rows,
        output_dir=str(target),
    )


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_merge_compatible_manifest(
    *,
    shard_path: Path,
    manifest: dict[str, Any],
    first_manifest: dict[str, Any],
) -> None:
    for key in ("dataset_name", "mode", "tasks", "label_source", "env_id"):
        if manifest.get(key) != first_manifest.get(key):
            raise ValueError(
                f"{shard_path} manifest {key} mismatch: "
                f"{manifest.get(key)!r} != {first_manifest.get(key)!r}"
            )


def _concat_jsonl_files(
    sources: Sequence[Path],
    target: Path,
    *,
    max_lines: int | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as output:
        for source in sources:
            if not source.exists():
                raise FileNotFoundError(str(source))
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if max_lines is not None and written >= max_lines:
                    return
                output.write(line + "\n")
                written += 1


def _merge_split_manifests(*, shard_paths: Sequence[Path], target: Path) -> None:
    merged: dict[str, set[int]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for shard_path in shard_paths:
        split_path = shard_path / "split_manifest.json"
        if not split_path.exists():
            continue
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for split_name in merged:
            values = payload.get(split_name, [])
            if isinstance(values, list):
                merged[split_name].update(int(value) for value in values)
    _write_json(
        target / "split_manifest.json",
        {split_name: sorted(values) for split_name, values in merged.items()},
    )


def write_pseudo_label_policy_dataset(
    *,
    dataset_name: str,
    mode: str,
    transitions: Iterable,
    action_manifest,
    game_metadata_by_id: dict[int, dict],
    splits: dict[str, set[int]],
    out_dir: str | Path,
    max_rows: int | None = None,
    tasks: tuple[str, ...] = ("policy_action",),
    token_budget: int = 2048,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1000,
    split_row_limits: SplitRowLimits | None = None,
) -> SftBuildResult:
    """Write rows from explicit high-confidence frame-derived action labels."""
    _validate_row_limit_contract(max_rows=max_rows, split_row_limits=split_row_limits)
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    rejection_reasons: Counter[str] = Counter()
    history = HistoryBuffer(max_items=16)
    accepted_policy_rows = 0
    accepted_next_frame_rows = 0
    processed_transitions = 0
    sample_rows: list[dict] = []
    accepted_rows_by_split = _empty_split_task_counts()
    split_quota_skips: Counter[str] = Counter()

    with ExitStack() as stack:
        handles: dict[tuple[str, str | None], Any] = {}
        for split_name in ("train", "validation", "test"):
            handles[(split_name, None)] = stack.enter_context(
                (target / f"{split_name}.jsonl").open("w", encoding="utf-8")
            )
            handles[(split_name, "policy_action")] = stack.enter_context(
                (target / f"{split_name}.policy_action.jsonl").open(
                    "w",
                    encoding="utf-8",
                )
            )
            handles[(split_name, "next_frame")] = stack.enter_context(
                (target / f"{split_name}.next_frame.jsonl").open(
                    "w",
                    encoding="utf-8",
                )
            )

        for transition in transitions:
            processed_transitions += 1
            if split_row_limits is not None and _all_split_limits_reached(
                tasks=tasks,
                counts=accepted_rows_by_split,
                limits=split_row_limits,
            ):
                break
            if (
                split_row_limits is None
                and max_rows is not None
                and _accepted_rows_for_limit(
                    tasks=tasks,
                    accepted_policy_rows=accepted_policy_rows,
                    accepted_next_frame_rows=accepted_next_frame_rows,
                )
                >= max_rows
            ):
                break
            split = _split_for_gameid(transition.gameid, splits)
            if split is None:
                rejection_reasons["gameid_not_in_split"] += 1
                _maybe_report_progress(
                    progress_callback=progress_callback,
                    progress_interval=progress_interval,
                    processed_transitions=processed_transitions,
                    accepted_policy_rows=accepted_policy_rows,
                    accepted_next_frame_rows=accepted_next_frame_rows,
                    rejected_rows=sum(rejection_reasons.values()),
                    reason="gameid_not_in_split",
                    last_gameid=transition.gameid,
                    last_step=transition.step,
                )
                continue
            write_policy = "policy_action" in tasks and _split_task_has_capacity(
                split=split,
                task="policy_action",
                counts=accepted_rows_by_split,
                limits=split_row_limits,
            )
            write_next_frame = "next_frame" in tasks and _split_task_has_capacity(
                split=split,
                task="next_frame",
                counts=accepted_rows_by_split,
                limits=split_row_limits,
            )
            if not write_policy and not write_next_frame:
                split_quota_skips[split] += 1
                continue
            pseudo_label = infer_visible_movement_pseudo_label(
                transition=transition,
                action_manifest=action_manifest,
            )
            if pseudo_label is None:
                rejection_reasons["pseudo_label_unavailable"] += 1
                _maybe_report_progress(
                    progress_callback=progress_callback,
                    progress_interval=progress_interval,
                    processed_transitions=processed_transitions,
                    accepted_policy_rows=accepted_policy_rows,
                    accepted_next_frame_rows=accepted_next_frame_rows,
                    rejected_rows=sum(rejection_reasons.values()),
                    reason="pseudo_label_unavailable",
                    last_gameid=transition.gameid,
                    last_step=transition.step,
                )
                continue
            game_metadata = game_metadata_by_id.get(transition.gameid, {})
            current_history = history.history_for(
                gameid=transition.gameid,
                mode=mode,
                token_budget=token_budget,
            )
            if write_policy:
                row = build_pseudo_policy_action_row(
                    dataset_name=dataset_name,
                    split=split,
                    mode=mode,
                    transition=transition,
                    action_manifest=action_manifest,
                    game_metadata=game_metadata,
                    history=current_history,
                    pseudo_label=pseudo_label,
                )
                _write_row(handles, row)
                if split == "train" and len(sample_rows) < 10:
                    sample_rows.append(row)
                accepted_policy_rows += 1
                accepted_rows_by_split[split]["policy_action"] += 1
            if write_next_frame:
                row = build_pseudo_next_frame_row(
                    dataset_name=dataset_name,
                    split=split,
                    mode=mode,
                    transition=transition,
                    action_manifest=action_manifest,
                    game_metadata=game_metadata,
                    history=current_history,
                    pseudo_label=pseudo_label,
                )
                _write_row(handles, row)
                if split == "train" and len(sample_rows) < 10:
                    sample_rows.append(row)
                accepted_next_frame_rows += 1
                accepted_rows_by_split[split]["next_frame"] += 1
            history.append(
                gameid=transition.gameid,
                observation_text=render_observation_text(transition.observation),
                action_id=pseudo_label.action_id,
                feedback=policy_feedback_from_outcome_observation(
                    action_id=pseudo_label.action_id,
                    observation=transition.next_observation,
                ),
            )
            _maybe_report_progress(
                progress_callback=progress_callback,
                progress_interval=progress_interval,
                processed_transitions=processed_transitions,
                accepted_policy_rows=accepted_policy_rows,
                accepted_next_frame_rows=accepted_next_frame_rows,
                rejected_rows=sum(rejection_reasons.values()),
                reason="accepted",
                last_gameid=transition.gameid,
                last_step=transition.step,
            )

    if "policy_action" in tasks and accepted_policy_rows == 0:
        raise RuntimeError(
            "pseudo-label dataset build produced zero policy_action rows"
        )
    if "next_frame" in tasks and accepted_next_frame_rows == 0:
        raise RuntimeError("pseudo-label dataset build produced zero next_frame rows")

    total_rejected = sum(rejection_reasons.values())
    split_limits_satisfied = _split_limits_satisfied(
        tasks=tasks,
        counts=accepted_rows_by_split,
        limits=split_row_limits,
    )
    manifest = {
        "schema_version": "learn-nethack.sft-manifest.v1",
        "dataset_name": dataset_name,
        "mode": mode,
        "tasks": list(tasks),
        "label_source": "pseudo_visible_player_delta",
        "true_keypress_label_available": False,
        "env_id": action_manifest.env_id,
        "accepted_policy_rows": accepted_policy_rows,
        "accepted_next_frame_rows": accepted_next_frame_rows,
        "accepted_rows_by_split": accepted_rows_by_split,
        "split_row_limits": (
            split_row_limits.as_dict() if split_row_limits is not None else None
        ),
        "split_limits_satisfied": split_limits_satisfied,
        "split_quota_skips": dict(sorted(split_quota_skips.items())),
        "rejected_rows": total_rejected,
        "next_frame_status": "ok" if accepted_next_frame_rows else "no_rows",
    }
    _write_json(target / "manifest.json", manifest)
    _write_json(
        target / "rejection_report.json",
        {
            "schema_version": "learn-nethack.sft-rejections.v1",
            "total_rejected": total_rejected,
            "reasons": dict(sorted(rejection_reasons.items())),
        },
    )
    action_manifest.save(target / "action_manifest.json")
    _write_json(
        target / "split_manifest.json",
        {split: sorted(gameids) for split, gameids in splits.items()},
    )
    with (target / "sample_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in sample_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    _raise_for_unsatisfied_split_limits(
        tasks=tasks,
        counts=accepted_rows_by_split,
        limits=split_row_limits,
    )

    return SftBuildResult(
        accepted_policy_rows=accepted_policy_rows,
        accepted_next_frame_rows=accepted_next_frame_rows,
        rejected_rows=total_rejected,
        output_dir=str(target),
    )


def _accepted_rows_for_limit(
    *,
    tasks: tuple[str, ...],
    accepted_policy_rows: int,
    accepted_next_frame_rows: int,
) -> int:
    if "policy_action" in tasks:
        return accepted_policy_rows
    if "next_frame" in tasks:
        return accepted_next_frame_rows
    return 0


def _maybe_report_progress(
    *,
    progress_callback: ProgressCallback | None,
    progress_interval: int,
    processed_transitions: int,
    accepted_policy_rows: int,
    accepted_next_frame_rows: int,
    rejected_rows: int,
    reason: str,
    last_gameid: int | None,
    last_step: int | None,
) -> None:
    if progress_callback is None:
        return
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    accepted_rows = accepted_policy_rows + accepted_next_frame_rows
    accepted_boundary = (
        reason == "accepted"
        and accepted_rows > 0
        and accepted_rows % progress_interval == 0
    )
    if processed_transitions % progress_interval != 0 and not accepted_boundary:
        return
    progress_callback(
        SftBuildProgress(
            processed_transitions=processed_transitions,
            accepted_policy_rows=accepted_policy_rows,
            accepted_next_frame_rows=accepted_next_frame_rows,
            rejected_rows=rejected_rows,
            reason=reason,
            last_gameid=None if last_gameid is None else int(last_gameid),
            last_step=None if last_step is None else int(last_step),
        )
    )


def build_sft_from_decoded_batches(
    *,
    dataset_name: str,
    mode: str,
    batches: Iterable[dict],
    action_manifest,
    gameids: list[int],
    game_metadata_by_id: dict[int, dict],
    out_dir: str | Path,
    max_rows: int | None,
    seed: int,
    tasks: tuple[str, ...] = ("policy_action", "next_frame"),
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1000,
    split_row_limits: SplitRowLimits | None = None,
    game_order_strategy: str = "source_order",
    token_budget: int = 2048,
) -> SftBuildResult:
    splits = split_gameids(gameids, seed=seed)
    split_sets = {
        "train": set(splits.train),
        "validation": set(splits.validation),
        "test": set(splits.test),
    }
    transitions = (
        transition for batch in batches for transition in normalize_decoded_batch(batch)
    )
    return write_sft_dataset(
        dataset_name=dataset_name,
        mode=mode,
        transitions=transitions,
        action_manifest=action_manifest,
        game_metadata_by_id=game_metadata_by_id,
        splits=split_sets,
        out_dir=out_dir,
        max_rows=max_rows,
        tasks=tasks,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        split_row_limits=split_row_limits,
        game_order_strategy=game_order_strategy,
        token_budget=token_budget,
    )


def build_pseudo_label_sft_from_frame_batches(
    *,
    dataset_name: str,
    mode: str,
    batches: Iterable[dict],
    action_manifest,
    gameids: list[int],
    game_metadata_by_id: dict[int, dict],
    out_dir: str | Path,
    max_rows: int | None,
    seed: int,
    tasks: tuple[str, ...] = ("policy_action",),
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 1000,
    split_row_limits: SplitRowLimits | None = None,
) -> SftBuildResult:
    """Build SFT rows from frame-only batches with explicit pseudo-labels."""
    splits = split_gameids(gameids, seed=seed)
    split_sets = {
        "train": set(splits.train),
        "validation": set(splits.validation),
        "test": set(splits.test),
    }
    transitions = (
        transition
        for batch in batches
        for transition in normalize_frame_only_batch(batch)
    )
    return write_pseudo_label_policy_dataset(
        dataset_name=dataset_name,
        mode=mode,
        transitions=transitions,
        action_manifest=action_manifest,
        game_metadata_by_id=game_metadata_by_id,
        splits=split_sets,
        out_dir=out_dir,
        max_rows=max_rows,
        tasks=tasks,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        split_row_limits=split_row_limits,
    )


def _validate_row_limit_contract(
    *,
    max_rows: int | None,
    split_row_limits: SplitRowLimits | None,
) -> None:
    if max_rows is not None and split_row_limits is not None:
        raise ValueError("use max_rows or split_row_limits, not both")
    if split_row_limits is None:
        return
    for split, limit in split_row_limits.as_dict().items():
        if limit <= 0:
            raise ValueError(f"split row limit for {split} must be positive")


def _empty_split_task_counts() -> dict[str, dict[str, int]]:
    return {
        split: {"policy_action": 0, "next_frame": 0}
        for split in ("train", "validation", "test")
    }


def _split_task_has_capacity(
    *,
    split: str,
    task: str,
    counts: Mapping[str, Mapping[str, int]],
    limits: SplitRowLimits | None,
) -> bool:
    if limits is None:
        return True
    return int(counts[split][task]) < int(limits.as_dict()[split])


def _all_split_limits_reached(
    *,
    tasks: tuple[str, ...],
    counts: Mapping[str, Mapping[str, int]],
    limits: SplitRowLimits,
) -> bool:
    limit_by_split = limits.as_dict()
    return all(
        int(counts[split][task]) >= int(limit_by_split[split])
        for split in ("train", "validation", "test")
        for task in tasks
    )


def _split_limits_satisfied(
    *,
    tasks: tuple[str, ...],
    counts: Mapping[str, Mapping[str, int]],
    limits: SplitRowLimits | None,
) -> bool | None:
    if limits is None:
        return None
    return _all_split_limits_reached(tasks=tasks, counts=counts, limits=limits)


def _raise_for_unsatisfied_split_limits(
    *,
    tasks: tuple[str, ...],
    counts: Mapping[str, Mapping[str, int]],
    limits: SplitRowLimits | None,
) -> None:
    if limits is None or _all_split_limits_reached(
        tasks=tasks,
        counts=counts,
        limits=limits,
    ):
        return
    actual = {
        split: {task: int(counts[split][task]) for task in tasks}
        for split in ("train", "validation", "test")
    }
    raise RuntimeError(
        "split row limits not satisfied: "
        f"expected={limits.as_dict()} per task; actual={actual}"
    )
