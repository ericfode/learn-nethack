from __future__ import annotations

from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import pytest

from learn_nethack.ttyrec import TTYREC_HEADER, write_terminal_ttyrec


def _read_ttyrec(path: Path) -> list[tuple[int, int, bytes]]:
    records: list[tuple[int, int, bytes]] = []
    with path.open("rb") as handle:
        while header := handle.read(TTYREC_HEADER.size):
            seconds, microseconds, payload_size = TTYREC_HEADER.unpack(header)
            payload = handle.read(payload_size)
            assert len(payload) == payload_size
            records.append((seconds, microseconds, payload))
    return records


def test_write_terminal_ttyrec_writes_deterministic_framed_records() -> None:
    with TemporaryDirectory() as tmp:
        target = write_terminal_ttyrec(
            Path(tmp) / "replay.ttyrec",
            ["MAP:\n@.", "MAP:\n.@"],
        )

        records = _read_ttyrec(target)

    assert [(seconds, microseconds) for seconds, microseconds, _ in records] == [
        (0, 0),
        (0, 250_000),
    ]
    assert records[0][2].startswith(b"\x1b[2J\x1b[HMAP:\r\n@.")
    assert records[1][2].endswith(b".@\r\n")


def test_write_terminal_ttyrec_rejects_nonpositive_frame_delay() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        write_terminal_ttyrec("unused.ttyrec", [], frame_delay_seconds=0)


def test_ttyrec_header_is_three_little_endian_unsigned_ints() -> None:
    assert TTYREC_HEADER.size == struct.calcsize("<III") == 12
