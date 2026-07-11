"""Deterministic ttyrec writing for rendered NetHack rollout frames."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import struct


TTYREC_HEADER = struct.Struct("<III")
DEFAULT_FRAME_DELAY_SECONDS = 0.25


def write_terminal_ttyrec(
    path: str | Path,
    frames: Iterable[str],
    *,
    frame_delay_seconds: float = DEFAULT_FRAME_DELAY_SECONDS,
) -> Path:
    """Write rendered terminal frames as a ttyrec v1-compatible stream."""
    if frame_delay_seconds <= 0:
        raise ValueError("frame_delay_seconds must be positive")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for frame_index, frame in enumerate(frames):
            timestamp = frame_index * frame_delay_seconds
            seconds = int(timestamp)
            microseconds = int(round((timestamp - seconds) * 1_000_000))
            if microseconds == 1_000_000:
                seconds += 1
                microseconds = 0
            terminal_text = "\x1b[2J\x1b[H" + str(frame).replace("\n", "\r\n")
            payload = (terminal_text + "\r\n").encode("utf-8")
            handle.write(TTYREC_HEADER.pack(seconds, microseconds, len(payload)))
            handle.write(payload)
    return target
