"""Deterministic rendering of NLE observations into text."""

from __future__ import annotations

from typing import Any


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _bytes_to_text(values: Any) -> str:
    values = _to_list(values) or []
    chars: list[str] = []
    for value in values:
        integer = int(value)
        if integer == 0:
            continue
        chars.append(chr(integer))
    return "".join(chars).strip()


def _grid_to_text(grid: Any, *, compact: bool = False) -> str:
    grid = _to_list(grid) or []
    lines: list[str] = []
    for row in grid:
        line = "".join(chr(int(value)) if int(value) else " " for value in row)
        lines.append(line.rstrip())
    if compact:
        lines = [line for line in lines if line.strip()]
    return "\n".join(lines).rstrip()


def observation_message_text(
    obs: dict[str, Any],
    *,
    max_message_chars: int = 240,
) -> str:
    """Return the visible NetHack message from an observation."""
    return _bytes_to_text(obs.get("message"))[:max_message_chars] or "<missing>"


def observation_blstat(obs: dict[str, Any], index: int) -> int | None:
    """Return one BLSTATS integer when available."""
    values = _to_list(obs.get("blstats"))
    if not isinstance(values, list) or len(values) <= index:
        return None
    try:
        return int(values[index])
    except (TypeError, ValueError):
        return None


def render_observation_text(
    obs: dict[str, Any],
    *,
    max_message_chars: int = 240,
    compact_map: bool = False,
) -> str:
    map_text = _grid_to_text(obs.get("tty_chars"), compact=compact_map) or "<missing>"
    message = observation_message_text(obs, max_message_chars=max_message_chars)
    blstats = _to_list(obs.get("blstats"))
    inventory = _to_list(obs.get("inventory"))

    blstats_text = str(blstats) if blstats else "<missing>"
    if inventory == []:
        inventory_text = "<empty>"
    elif inventory:
        inventory_text = str(inventory)
    else:
        inventory_text = "<missing>"

    return "\n".join(
        [
            "MAP:",
            map_text,
            "MESSAGE:",
            message,
            "BLSTATS:",
            blstats_text,
            "INVENTORY:",
            inventory_text,
        ]
    )
