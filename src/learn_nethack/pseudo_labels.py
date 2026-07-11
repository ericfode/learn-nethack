"""Pseudo-label inference for frame-only NetHack ttyrec transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PLAYER_GLYPH = 64


@dataclass(frozen=True)
class PseudoActionLabel:
    action_id: int
    action_name: str
    key_label: str
    direction: str
    confidence: float
    label_source: str
    reason: str


_DIRECTION_BY_DELTA = {
    (-1, 0): "N",
    (-1, 1): "NE",
    (0, 1): "E",
    (1, 1): "SE",
    (1, 0): "S",
    (1, -1): "SW",
    (0, -1): "W",
    (-1, -1): "NW",
}


def infer_visible_movement_pseudo_label(
    *,
    transition: Any,
    action_manifest: Any,
) -> PseudoActionLabel | None:
    """Infer a high-confidence movement label from visible player-frame deltas."""
    if transition.next_observation is None:
        return None
    current_positions = _player_positions(transition.observation)
    next_positions = _player_positions(transition.next_observation)
    if len(current_positions) != 1 or len(next_positions) != 1:
        return None
    current_row, current_col = current_positions[0]
    next_row, next_col = next_positions[0]
    delta = (next_row - current_row, next_col - current_col)
    direction = _DIRECTION_BY_DELTA.get(delta)
    if direction is None:
        return None
    entry = _manifest_entry_for_compass_direction(action_manifest, direction)
    if entry is None:
        return None
    return PseudoActionLabel(
        action_id=int(entry.action_id),
        action_name=str(entry.nle_action_name),
        key_label=str(entry.key_label),
        direction=direction,
        confidence=1.0,
        label_source="pseudo_visible_player_delta",
        reason="single_visible_player_moved_one_cell",
    )


def _player_positions(observation: dict[str, Any]) -> list[tuple[int, int]]:
    tty_chars = observation.get("tty_chars") or []
    positions: list[tuple[int, int]] = []
    for row_index, row in enumerate(tty_chars):
        start = 0
        while start < len(row):
            try:
                col_index = row.index(PLAYER_GLYPH, start)
            except ValueError:
                break
            positions.append((row_index, col_index))
            if len(positions) > 1:
                return positions
            start = col_index + 1
    return positions


def _manifest_entry_for_compass_direction(
    action_manifest: Any,
    direction: str,
) -> Any | None:
    exact_name = f"CompassDirection.{direction}"
    for entry in action_manifest.entries:
        if str(entry.nle_action_name) == exact_name:
            return entry
    return None
