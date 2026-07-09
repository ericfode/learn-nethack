"""Build compact true-action terminal transition arrays from NLD."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from learn_nethack.action_manifest import ActionManifest
from learn_nethack.nld_decode import normalize_decoded_batch
from learn_nethack.nld_metadata import inspect_nld_db, read_gameids, split_gameids
from learn_nethack.world_model_ingest import (
    SPLIT_CODES,
    ambiguous_raw_keys,
    collect_selected_games_parallel,
    terminal_plane,
    unambiguous_raw_key_mapping,
)
from learn_nethack.world_model_metrics import TERMINAL_COLS, TERMINAL_ROWS


@dataclass(frozen=True)
class LocalWorldModelDataConfig:
    seed: int = 20260709
    train_games: int = 48
    validation_games: int = 12
    test_games: int = 12
    train_transitions: int = 12_000
    validation_transitions: int = 2_000
    test_transitions: int = 2_000
    decoder_workers: int = 12


@dataclass(frozen=True)
class LocalWorldModelDataResult:
    schema_version: str
    dataset_path: str
    manifest_path: str
    counts: dict[str, int]
    rejection_reasons: dict[str, int]


def build_local_world_model_dataset(
    *,
    db_path: str | Path,
    action_manifest_path: str | Path,
    out_dir: str | Path,
    config: LocalWorldModelDataConfig,
) -> LocalWorldModelDataResult:
    """Decode selected disjoint games and write a compressed transition array."""
    source = Path(db_path).resolve()
    action_manifest_source = Path(action_manifest_path).resolve()
    target = Path(out_dir)
    report = inspect_nld_db(source)
    all_gameids = read_gameids(source)
    split = split_gameids(all_gameids, seed=config.seed)
    selected = {
        "train": _select_gameids(split.train, config.train_games, config.seed, "train"),
        "validation": _select_gameids(
            split.validation,
            config.validation_games,
            config.seed,
            "validation",
        ),
        "test": _select_gameids(split.test, config.test_games, config.seed, "test"),
    }
    if config.decoder_workers <= 0:
        raise ValueError("decoder_workers must be positive")
    arrays, counts, rejections = collect_selected_games_parallel(
        dataset_name=report.dataset_name,
        db_path=source,
        action_manifest_path=action_manifest_source,
        selected_gameids=selected,
        transition_limits={
            "train": config.train_transitions,
            "validation": config.validation_transitions,
            "test": config.test_transitions,
        },
        decoder_workers=config.decoder_workers,
    )
    for split_name, requested in (
        ("train", config.train_transitions),
        ("validation", config.validation_transitions),
        ("test", config.test_transitions),
    ):
        if counts[split_name] < requested:
            raise RuntimeError(
                f"NLD extraction produced {counts[split_name]} {split_name} "
                f"transitions; requested {requested}"
            )

    target.mkdir(parents=True, exist_ok=True)
    dataset_path = target / "transitions.npz"
    _write_npz(dataset_path, arrays)
    manifest = {
        "schema_version": "learn-nethack.local-world-model-data.v1",
        "source": {
            "db_path": str(source),
            "dataset_name": report.dataset_name,
            "ttyrec_version": report.ttyrec_version,
            "action_manifest_path": str(action_manifest_source),
        },
        "config": asdict(config),
        "split_codes": SPLIT_CODES,
        "selected_gameids": selected,
        "counts": counts,
        "rejection_reasons": dict(sorted(rejections.items())),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "dataset_sha256": _sha256(dataset_path),
    }
    manifest_path = target / "manifest.json"
    _write_json(manifest_path, manifest)
    return LocalWorldModelDataResult(
        schema_version="learn-nethack.local-world-model-data-result.v1",
        dataset_path=str(dataset_path),
        manifest_path=str(manifest_path),
        counts=counts,
        rejection_reasons=dict(sorted(rejections.items())),
    )


def collect_terminal_transition_arrays(
    *,
    batches: Iterable[dict[str, Any]],
    action_manifest: ActionManifest,
    split_gameids_by_name: dict[str, set[int]],
    transition_limits: dict[str, int],
):
    """Collect bounded transition arrays while preserving trace continuity."""
    import numpy as np

    _validate_split_inputs(split_gameids_by_name, transition_limits)
    capacity = sum(transition_limits.values())
    arrays = {
        "current_chars": np.empty(
            (capacity, TERMINAL_ROWS, TERMINAL_COLS), dtype=np.uint8
        ),
        "current_colors": np.empty(
            (capacity, TERMINAL_ROWS, TERMINAL_COLS), dtype=np.uint8
        ),
        "next_chars": np.empty(
            (capacity, TERMINAL_ROWS, TERMINAL_COLS), dtype=np.uint8
        ),
        "next_colors": np.empty(
            (capacity, TERMINAL_ROWS, TERMINAL_COLS), dtype=np.uint8
        ),
        "action_ids": np.empty(capacity, dtype=np.uint16),
        "raw_key_codes": np.empty(capacity, dtype=np.int16),
        "gameids": np.empty(capacity, dtype=np.int32),
        "sequence_ids": np.empty(capacity, dtype=np.int32),
        "sequence_steps": np.empty(capacity, dtype=np.int32),
        "split_codes": np.empty(capacity, dtype=np.uint8),
    }
    counts = {name: 0 for name in SPLIT_CODES}
    rejections: Counter[str] = Counter()
    raw_key_mapping = unambiguous_raw_key_mapping(action_manifest)
    ambiguous_keys = ambiguous_raw_keys(action_manifest)
    sequence_numbers: dict[str, int] = {}
    offset = 0
    fallback_sequence = 0

    for batch_index, batch in enumerate(batches):
        for transition in normalize_decoded_batch(batch):
            split_name = _split_name(transition.gameid, split_gameids_by_name)
            if split_name is None:
                rejections["gameid_not_selected"] += 1
                continue
            if counts[split_name] >= transition_limits[split_name]:
                continue
            if transition.next_observation is None:
                rejections["missing_next_observation"] += 1
                continue
            raw_key_code = int(transition.raw_key_code)
            if raw_key_code in ambiguous_keys:
                rejections["ambiguous_raw_key_code"] += 1
                continue
            action_id = raw_key_mapping.get(raw_key_code)
            if action_id is None:
                rejections["unmapped_raw_key_code"] += 1
                continue
            try:
                current_chars = terminal_plane(
                    transition.observation.get("tty_chars"),
                    name="current_chars",
                    max_value=255,
                )
                current_colors = terminal_plane(
                    transition.observation.get("tty_colors"),
                    name="current_colors",
                    max_value=31,
                )
                next_chars = terminal_plane(
                    transition.next_observation.get("tty_chars"),
                    name="next_chars",
                    max_value=255,
                )
                next_colors = terminal_plane(
                    transition.next_observation.get("tty_colors"),
                    name="next_colors",
                    max_value=31,
                )
            except ValueError as exc:
                rejections[str(exc)] += 1
                continue

            sequence_key = transition.sequence_id
            if sequence_key is None:
                sequence_key = (
                    f"fallback:{transition.gameid}:{batch_index}:{fallback_sequence}"
                )
                fallback_sequence += 1
            sequence_number = sequence_numbers.setdefault(
                sequence_key,
                len(sequence_numbers),
            )
            sequence_step = (
                int(transition.sequence_step)
                if transition.sequence_step is not None
                else int(transition.step)
            )
            arrays["current_chars"][offset] = current_chars
            arrays["current_colors"][offset] = current_colors
            arrays["next_chars"][offset] = next_chars
            arrays["next_colors"][offset] = next_colors
            arrays["action_ids"][offset] = action_id
            arrays["raw_key_codes"][offset] = raw_key_code
            arrays["gameids"][offset] = transition.gameid
            arrays["sequence_ids"][offset] = sequence_number
            arrays["sequence_steps"][offset] = sequence_step
            arrays["split_codes"][offset] = SPLIT_CODES[split_name]
            counts[split_name] += 1
            offset += 1
        if all(counts[name] >= transition_limits[name] for name in SPLIT_CODES):
            break

    return (
        {name: value[:offset] for name, value in arrays.items()},
        counts,
        rejections,
    )


def load_transition_arrays(path: str | Path) -> dict[str, Any]:
    """Load all arrays into memory and validate their row counts."""
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    required = {
        "current_chars",
        "current_colors",
        "next_chars",
        "next_colors",
        "action_ids",
        "gameids",
        "sequence_ids",
        "sequence_steps",
        "split_codes",
    }
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"transition dataset is missing arrays: {sorted(missing)}")
    row_counts = {name: int(value.shape[0]) for name, value in arrays.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"transition array row counts differ: {row_counts}")
    return arrays


def _select_gameids(
    gameids: list[int],
    count: int,
    seed: int,
    split_name: str,
) -> list[int]:
    if count <= 0:
        raise ValueError(f"{split_name}_games must be positive")
    ranked = sorted(
        gameids,
        key=lambda gameid: hashlib.sha256(
            f"{seed}:{split_name}:{gameid}".encode("utf-8")
        ).digest(),
    )
    selected = sorted(ranked[:count])
    if len(selected) < count:
        raise RuntimeError(
            f"split {split_name} has {len(selected)} games; requested {count}"
        )
    return selected


def _split_name(
    gameid: int,
    split_gameids_by_name: dict[str, set[int]],
) -> str | None:
    for name in ("train", "validation", "test"):
        if gameid in split_gameids_by_name[name]:
            return name
    return None


def _validate_split_inputs(
    split_gameids_by_name: dict[str, set[int]],
    transition_limits: dict[str, int],
) -> None:
    if set(split_gameids_by_name) != set(SPLIT_CODES):
        raise ValueError("split_gameids_by_name must contain train, validation, test")
    if set(transition_limits) != set(SPLIT_CODES):
        raise ValueError("transition_limits must contain train, validation, test")
    seen: set[int] = set()
    for name in SPLIT_CODES:
        if transition_limits[name] <= 0:
            raise ValueError(f"{name} transition limit must be positive")
        overlap = seen & split_gameids_by_name[name]
        if overlap:
            raise ValueError(f"game IDs overlap across splits: {sorted(overlap)}")
        seen.update(split_gameids_by_name[name])


def _write_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    np.savez_compressed(path, **arrays)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
