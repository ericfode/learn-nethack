"""Adapters for decoded NLD ttyrec batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DecodedTransition:
    gameid: int
    step: int
    raw_key_code: int
    observation: dict[str, Any]
    next_observation: dict[str, Any] | None
    sequence_id: str | None = None
    sequence_step: int | None = None


@dataclass(frozen=True)
class DecodedFrameTransition:
    gameid: int
    step: int
    observation: dict[str, Any]
    next_observation: dict[str, Any] | None
    sequence_id: str | None = None
    sequence_step: int | None = None


def _list_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_sequence_batch(gameids: Any) -> bool:
    return bool(gameids) and isinstance(gameids[0], list)


def _value_at(value: Any, coordinates: tuple[int, ...], default: Any) -> Any:
    if value is None:
        return default
    current = value
    for coordinate in coordinates:
        current = current[coordinate]
    return current


def _optional_value(
    batch: dict[str, Any], key: str, coordinates: tuple[int, ...], default: Any
) -> Any:
    if key not in batch:
        return default
    return _value_at(_list_value(batch[key]), coordinates, default)


def _timestamp_sequence_id(
    batch: dict[str, Any],
    *,
    gameid: int,
    coordinates: tuple[int, ...],
) -> str | None:
    if "timestamps" not in batch:
        return None
    timestamp = _optional_value(batch, "timestamps", coordinates, None)
    if timestamp is None:
        return None
    return f"{int(gameid)}:{int(timestamp)}"


def _observation_at(
    batch: dict[str, Any], coordinates: tuple[int, ...]
) -> dict[str, Any]:
    observation = {
        "tty_chars": _optional_value(batch, "tty_chars", coordinates, []),
        "message": _optional_value(batch, "message", coordinates, []),
        "blstats": _optional_value(batch, "blstats", coordinates, []),
        "inventory": _optional_value(batch, "inventory", coordinates, []),
    }
    for key in ("tty_colors", "tty_cursor", "scores", "timestamps", "done"):
        if key in batch:
            observation[key] = _optional_value(batch, key, coordinates, None)
    return observation


def _transition_done(batch: dict[str, Any], coordinates: tuple[int, ...]) -> bool:
    return bool(_optional_value(batch, "done", coordinates, False))


def _normalize_sequence_batch(batch: dict[str, Any]) -> Iterable[DecodedTransition]:
    gameids = _list_value(batch["gameids"])
    steps = _list_value(batch.get("steps")) if "steps" in batch else None
    keypresses = _list_value(batch.get("keypresses", batch.get("actions")))
    if keypresses is None:
        raise ValueError(
            "decoded NLD batch has no keypresses or actions field; "
            f"available keys: {', '.join(sorted(batch))}"
        )

    for batch_index, gameid_sequence in enumerate(gameids):
        first_gameid = int(gameid_sequence[0])
        sequence_id = _timestamp_sequence_id(
            batch,
            gameid=first_gameid,
            coordinates=(batch_index, 0),
        )
        for time_index, gameid in enumerate(gameid_sequence):
            coordinates = (batch_index, time_index)
            next_observation = None
            if (
                not _transition_done(batch, coordinates)
                and time_index + 1 < len(gameid_sequence)
                and int(gameid_sequence[time_index + 1]) == int(gameid)
            ):
                next_observation = _observation_at(batch, (batch_index, time_index + 1))

            step = (
                _value_at(steps, coordinates, time_index)
                if steps is not None
                else time_index
            )
            yield DecodedTransition(
                gameid=int(gameid),
                step=int(step),
                raw_key_code=int(_value_at(keypresses, coordinates, 0)),
                observation=_observation_at(batch, coordinates),
                next_observation=next_observation,
                sequence_id=sequence_id,
                sequence_step=time_index,
            )


def _normalize_flat_batch(batch: dict[str, Any]) -> Iterable[DecodedTransition]:
    gameids = _list_value(batch["gameids"])
    steps = _list_value(batch.get("steps", list(range(len(gameids)))))
    keypresses = _list_value(batch.get("keypresses", batch.get("actions")))
    if keypresses is None:
        raise ValueError(
            "decoded NLD batch has no keypresses or actions field; "
            f"available keys: {', '.join(sorted(batch))}"
        )

    observations: list[dict[str, Any]] = []
    for index, _gameid in enumerate(gameids):
        observations.append(_observation_at(batch, (index,)))

    for index, gameid in enumerate(gameids):
        next_observation = None
        if (
            not _transition_done(batch, (index,))
            and index + 1 < len(gameids)
            and int(gameids[index + 1]) == int(gameid)
        ):
            next_observation = observations[index + 1]
        yield DecodedTransition(
            gameid=int(gameid),
            step=int(steps[index]),
            raw_key_code=int(keypresses[index]),
            observation=observations[index],
            next_observation=next_observation,
        )


def normalize_decoded_batch(batch: dict[str, Any]) -> Iterable[DecodedTransition]:
    gameids = _list_value(batch["gameids"])
    if _is_sequence_batch(gameids):
        yield from _normalize_sequence_batch(batch)
    else:
        yield from _normalize_flat_batch(batch)


def normalize_frame_only_batch(
    batch: dict[str, Any],
) -> Iterable[DecodedFrameTransition]:
    """Pair frame observations from decoded batches that do not expose keypresses."""
    gameids = _list_value(batch["gameids"])
    if _is_sequence_batch(gameids):
        yield from _normalize_frame_only_sequence_batch(batch)
    else:
        yield from _normalize_frame_only_flat_batch(batch)


def _normalize_frame_only_sequence_batch(
    batch: dict[str, Any],
) -> Iterable[DecodedFrameTransition]:
    gameids = _list_value(batch["gameids"])
    steps = _list_value(batch.get("steps")) if "steps" in batch else None
    for batch_index, gameid_sequence in enumerate(gameids):
        first_gameid = int(gameid_sequence[0])
        sequence_id = _timestamp_sequence_id(
            batch,
            gameid=first_gameid,
            coordinates=(batch_index, 0),
        )
        for time_index, gameid in enumerate(gameid_sequence):
            coordinates = (batch_index, time_index)
            next_observation = None
            if (
                not _transition_done(batch, coordinates)
                and time_index + 1 < len(gameid_sequence)
                and int(gameid_sequence[time_index + 1]) == int(gameid)
            ):
                next_observation = _observation_at(batch, (batch_index, time_index + 1))
            step = (
                _value_at(steps, coordinates, time_index)
                if steps is not None
                else time_index
            )
            yield DecodedFrameTransition(
                gameid=int(gameid),
                step=int(step),
                observation=_observation_at(batch, coordinates),
                next_observation=next_observation,
                sequence_id=sequence_id,
                sequence_step=time_index,
            )


def _normalize_frame_only_flat_batch(
    batch: dict[str, Any],
) -> Iterable[DecodedFrameTransition]:
    gameids = _list_value(batch["gameids"])
    steps = _list_value(batch.get("steps", list(range(len(gameids)))))
    observations = [_observation_at(batch, (index,)) for index, _ in enumerate(gameids)]
    for index, gameid in enumerate(gameids):
        next_observation = None
        if (
            not _transition_done(batch, (index,))
            and index + 1 < len(gameids)
            and int(gameids[index + 1]) == int(gameid)
        ):
            next_observation = observations[index + 1]
        yield DecodedFrameTransition(
            gameid=int(gameid),
            step=int(steps[index]),
            observation=observations[index],
            next_observation=next_observation,
        )


def iter_nld_ttyrec_batches(
    *,
    dataset_name: str,
    batch_size: int,
    seq_length: int = 32,
    rows: int = 24,
    cols: int = 80,
    dbfilename: str = "ttyrecs.db",
    gameids: list[int] | None = None,
    shuffle: bool = True,
    loop_forever: bool = False,
):
    """Yield batches from nle.dataset without importing NLE at module import time."""
    try:
        import nle.dataset as nld
    except ImportError as exc:  # pragma: no cover - depends on optional NLE install.
        raise RuntimeError("nle.dataset is required for NLD ttyrec decoding") from exc

    dataset = nld.TtyrecDataset(
        dataset_name,
        batch_size=batch_size,
        seq_length=seq_length,
        rows=rows,
        cols=cols,
        dbfilename=dbfilename,
        gameids=gameids,
        shuffle=shuffle,
        loop_forever=loop_forever,
    )
    for batch in dataset:
        yield batch
