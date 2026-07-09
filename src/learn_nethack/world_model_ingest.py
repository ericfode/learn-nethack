"""Process-isolated NLD game decoding for local world-model datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing
from pathlib import Path
from typing import Any

from learn_nethack.action_manifest import ActionManifest, load_action_manifest
from learn_nethack.nld_decode import iter_nld_ttyrec_batches, normalize_decoded_batch
from learn_nethack.world_model_metrics import TERMINAL_COLS, TERMINAL_ROWS


SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


def collect_selected_games_parallel(
    *,
    dataset_name: str,
    db_path: Path,
    action_manifest_path: Path,
    selected_gameids: dict[str, list[int]],
    transition_limits: dict[str, int],
    decoder_workers: int,
):
    """Decode independent games in processes so NLE's converter can use cores."""
    tasks: list[tuple[str, int, int]] = []
    for split_name in ("train", "validation", "test"):
        gameids = selected_gameids[split_name]
        raw_limit = math.ceil(transition_limits[split_name] / len(gameids) * 1.35) + 16
        tasks.extend((split_name, gameid, raw_limit) for gameid in gameids)

    results: dict[tuple[str, int], dict[str, Any]] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=decoder_workers,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(
                _decode_game_worker,
                dataset_name=dataset_name,
                db_path=str(db_path),
                action_manifest_path=str(action_manifest_path),
                split_name=split_name,
                gameid=gameid,
                raw_transition_limit=raw_limit,
            ): (split_name, gameid)
            for split_name, gameid, raw_limit in tasks
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    return _merge_game_worker_results(
        results=results,
        selected_gameids=selected_gameids,
        transition_limits=transition_limits,
    )


def terminal_plane(value: Any, *, name: str, max_value: int):
    import numpy as np

    array = np.asarray(value)
    if array.shape != (TERMINAL_ROWS, TERMINAL_COLS):
        raise ValueError(f"invalid_{name}_shape")
    if array.size and (int(array.min()) < 0 or int(array.max()) > max_value):
        raise ValueError(f"invalid_{name}_value")
    return array.astype(np.uint8, copy=False)


def unambiguous_raw_key_mapping(manifest: ActionManifest) -> dict[int, int]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for entry in manifest.entries:
        grouped[int(entry.raw_key_code)].append(int(entry.action_id))
    return {raw: values[0] for raw, values in grouped.items() if len(values) == 1}


def ambiguous_raw_keys(manifest: ActionManifest) -> set[int]:
    grouped: Counter[int] = Counter(
        int(entry.raw_key_code) for entry in manifest.entries
    )
    return {raw for raw, count in grouped.items() if count > 1}


def _decode_game_worker(
    *,
    dataset_name: str,
    db_path: str,
    action_manifest_path: str,
    split_name: str,
    gameid: int,
    raw_transition_limit: int,
) -> dict[str, Any]:
    """Decode a bounded prefix of one game in an isolated process."""
    import numpy as np

    manifest = load_action_manifest(action_manifest_path)
    raw_key_mapping = unambiguous_raw_key_mapping(manifest)
    ambiguous_keys = ambiguous_raw_keys(manifest)
    batches = iter_nld_ttyrec_batches(
        dataset_name=dataset_name,
        batch_size=1,
        seq_length=raw_transition_limit + 1,
        dbfilename=db_path,
        gameids=[gameid],
        shuffle=False,
        loop_forever=False,
    )
    try:
        batch = next(iter(batches))
    except StopIteration as exc:
        raise RuntimeError(f"NLD produced no batch for game {gameid}") from exc

    values: dict[str, list[Any]] = {
        "current_chars": [],
        "current_colors": [],
        "next_chars": [],
        "next_colors": [],
        "action_ids": [],
        "raw_key_codes": [],
        "gameids": [],
        "sequence_steps": [],
        "split_codes": [],
    }
    rejections: Counter[str] = Counter()
    for transition in normalize_decoded_batch(batch):
        if transition.gameid != gameid:
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
        values["current_chars"].append(current_chars)
        values["current_colors"].append(current_colors)
        values["next_chars"].append(next_chars)
        values["next_colors"].append(next_colors)
        values["action_ids"].append(action_id)
        values["raw_key_codes"].append(raw_key_code)
        values["gameids"].append(gameid)
        values["sequence_steps"].append(
            int(transition.sequence_step)
            if transition.sequence_step is not None
            else int(transition.step)
        )
        values["split_codes"].append(SPLIT_CODES[split_name])

    count = len(values["action_ids"])
    return {
        "split_name": split_name,
        "gameid": gameid,
        "count": count,
        "rejection_reasons": dict(rejections),
        "arrays": {
            "current_chars": np.asarray(values["current_chars"], dtype=np.uint8),
            "current_colors": np.asarray(values["current_colors"], dtype=np.uint8),
            "next_chars": np.asarray(values["next_chars"], dtype=np.uint8),
            "next_colors": np.asarray(values["next_colors"], dtype=np.uint8),
            "action_ids": np.asarray(values["action_ids"], dtype=np.uint16),
            "raw_key_codes": np.asarray(values["raw_key_codes"], dtype=np.int16),
            "gameids": np.asarray(values["gameids"], dtype=np.int32),
            "sequence_steps": np.asarray(values["sequence_steps"], dtype=np.int32),
            "split_codes": np.asarray(values["split_codes"], dtype=np.uint8),
        },
    }


def _merge_game_worker_results(
    *,
    results: dict[tuple[str, int], dict[str, Any]],
    selected_gameids: dict[str, list[int]],
    transition_limits: dict[str, int],
):
    import numpy as np

    names = (
        "current_chars",
        "current_colors",
        "next_chars",
        "next_colors",
        "action_ids",
        "raw_key_codes",
        "gameids",
        "sequence_steps",
        "split_codes",
    )
    pieces: dict[str, list[Any]] = {name: [] for name in names}
    sequence_pieces: list[Any] = []
    counts: dict[str, int] = {}
    rejections: Counter[str] = Counter()
    sequence_number = 0
    for split_name in ("train", "validation", "test"):
        remaining = transition_limits[split_name]
        for gameid in sorted(selected_gameids[split_name]):
            result = results[(split_name, gameid)]
            take = min(int(result["count"]), remaining)
            for reason, count in result["rejection_reasons"].items():
                rejections[str(reason)] += int(count)
            if take <= 0:
                sequence_number += 1
                continue
            for name in names:
                pieces[name].append(result["arrays"][name][:take])
            sequence_pieces.append(np.full(take, sequence_number, dtype=np.int32))
            sequence_number += 1
            remaining -= take
            if remaining == 0:
                break
        counts[split_name] = transition_limits[split_name] - remaining

    arrays = {name: np.concatenate(values) for name, values in pieces.items()}
    arrays["sequence_ids"] = np.concatenate(sequence_pieces)
    return arrays, counts, rejections
