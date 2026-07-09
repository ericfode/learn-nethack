"""Archive-backed NLD shard readers for Modal-friendly full-data builds."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Iterator

from learn_nethack.nld_decode import iter_nld_ttyrec_batches


BatchIterator = Callable[..., Iterable[dict[str, Any]]]


@dataclass(frozen=True)
class ArchiveShard:
    shard_tar: str
    shard_db: str


@dataclass(frozen=True)
class ArchiveDatasetPlan:
    dataset_name: str
    shards: tuple[ArchiveShard, ...]
    gameids: tuple[int, ...]
    gameids_by_shard: dict[ArchiveShard, tuple[int, ...]]
    game_metadata_by_id: dict[int, dict[str, Any]]

    def gameids_for(self, shard: ArchiveShard) -> tuple[int, ...]:
        return self.gameids_by_shard[shard]


def read_archive_manifest(path: str | Path) -> list[ArchiveShard]:
    """Read a JSON or JSONL manifest of tar-shard and sidecar-DB pairs."""
    manifest_path = Path(path)
    raw = manifest_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if raw.startswith("["):
        payload = json.loads(raw)
        records = payload
    elif raw.startswith("{") and "\n" not in raw:
        payload = json.loads(raw)
        records = payload["shards"] if "shards" in payload else [payload]
    else:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    shards: list[ArchiveShard] = []
    for record in records:
        if "shard_tar" not in record or "shard_db" not in record:
            raise ValueError("archive manifest rows require shard_tar and shard_db")
        shards.append(
            ArchiveShard(
                shard_tar=str(
                    _resolve_manifest_path(manifest_path, record["shard_tar"])
                ),
                shard_db=str(_resolve_manifest_path(manifest_path, record["shard_db"])),
            )
        )
    return shards


def plan_archive_dataset(path: str | Path) -> ArchiveDatasetPlan:
    """Read sidecar DBs and build one episode-safe archive dataset plan."""
    from learn_nethack.nld_metadata import (
        inspect_nld_db,
        read_game_metadata,
        read_gameids,
    )

    shards = tuple(read_archive_manifest(path))
    if not shards:
        raise RuntimeError("archive manifest contains no shards")

    dataset_name: str | None = None
    gameids_by_shard: dict[ArchiveShard, tuple[int, ...]] = {}
    game_metadata_by_id: dict[int, dict[str, Any]] = {}
    seen_gameids: set[int] = set()
    for shard in shards:
        report = inspect_nld_db(shard.shard_db)
        if dataset_name is None:
            dataset_name = report.dataset_name
        elif report.dataset_name != dataset_name:
            raise ValueError(
                f"archive shard dataset_name mismatch: "
                f"{report.dataset_name!r} != {dataset_name!r}"
            )

        shard_gameids = tuple(read_gameids(shard.shard_db))
        seen_gameids.update(shard_gameids)
        gameids_by_shard[shard] = shard_gameids
        game_metadata_by_id.update(read_game_metadata(shard.shard_db))

    if dataset_name is None:
        raise RuntimeError("archive manifest contains no readable shard DBs")

    return ArchiveDatasetPlan(
        dataset_name=dataset_name,
        shards=shards,
        gameids=tuple(sorted(seen_gameids)),
        gameids_by_shard=gameids_by_shard,
        game_metadata_by_id=game_metadata_by_id,
    )


def iter_archive_dataset_batches(
    plan: ArchiveDatasetPlan,
    *,
    selected_gameids: Iterable[int] | None = None,
    shard_indices: Iterable[int] | None = None,
    batch_size: int,
    seq_length: int = 32,
    rows: int = 24,
    cols: int = 80,
    shuffle: bool = False,
    loop_forever: bool = False,
    batch_iterator: BatchIterator = iter_nld_ttyrec_batches,
) -> Iterator[dict[str, Any]]:
    """Yield decoded batches across archive shards, routing selected gameids."""
    selected_set = None if selected_gameids is None else set(selected_gameids)
    selected_shard_indices = (
        None if shard_indices is None else {int(index) for index in shard_indices}
    )
    if selected_shard_indices is not None:
        invalid = sorted(
            index
            for index in selected_shard_indices
            if index < 0 or index >= len(plan.shards)
        )
        if invalid:
            raise IndexError(f"archive shard index out of range: {invalid[0]}")
    for shard_index, shard in enumerate(plan.shards):
        if (
            selected_shard_indices is not None
            and shard_index not in selected_shard_indices
        ):
            continue
        shard_gameids = list(plan.gameids_for(shard))
        if selected_set is not None:
            shard_gameids = [
                gameid for gameid in shard_gameids if gameid in selected_set
            ]
        if not shard_gameids:
            continue
        yield from iter_nld_archive_shard_batches(
            shard,
            dataset_name=plan.dataset_name,
            batch_size=batch_size,
            seq_length=seq_length,
            rows=rows,
            cols=cols,
            gameids=shard_gameids,
            shuffle=shuffle,
            loop_forever=loop_forever,
            batch_iterator=batch_iterator,
        )


def iter_nld_archive_shard_batches(
    shard: ArchiveShard,
    *,
    dataset_name: str,
    batch_size: int,
    seq_length: int = 32,
    rows: int = 24,
    cols: int = 80,
    gameids: list[int] | None = None,
    shuffle: bool = False,
    loop_forever: bool = False,
    batch_iterator: BatchIterator = iter_nld_ttyrec_batches,
) -> Iterator[dict[str, Any]]:
    """Extract one tar shard to temp storage and yield decoded NLD batches."""
    from learn_nethack.modal_upload import safe_extract_tar_shard
    from learn_nethack.nld_metadata import copy_db_with_rewritten_root

    with TemporaryDirectory(prefix="learn-nethack-nld-shard-") as tmp:
        temp_root = Path(tmp)
        extract_root = temp_root / "nld-shard"
        staged_db = temp_root / "ttyrecs.db"
        safe_extract_tar_shard(
            shard_path=shard.shard_tar,
            destination_root=extract_root,
        )
        copy_db_with_rewritten_root(
            source_db=shard.shard_db,
            target_db=staged_db,
            new_root=str(extract_root),
        )
        yield from batch_iterator(
            dataset_name=dataset_name,
            batch_size=batch_size,
            seq_length=seq_length,
            rows=rows,
            cols=cols,
            dbfilename=str(staged_db),
            gameids=gameids,
            shuffle=shuffle,
            loop_forever=loop_forever,
        )


def _resolve_manifest_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return manifest_path.parent / path
