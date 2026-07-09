"""Index local alt.org/NAO ttyrec trees into NLE dataset databases."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def prepare_altorg_staging_root(
    *,
    metadata_root: str | Path,
    ttyrec_root: str | Path,
    staging_root: str | Path,
) -> dict[str, Any]:
    """Create a canonical alt.org directory view using symlinks."""
    metadata = Path(metadata_root)
    ttyrecs = Path(ttyrec_root)
    staging = Path(staging_root)
    if not metadata.exists():
        raise FileNotFoundError(f"metadata_root does not exist: {metadata}")
    if not ttyrecs.exists():
        raise FileNotFoundError(f"ttyrec_root does not exist: {ttyrecs}")

    xlogfiles = sorted(metadata.glob("xlogfile.*"))
    if not xlogfiles:
        raise ValueError(f"metadata_root has no xlogfile.* files: {metadata}")
    blacklist = metadata / "blacklist.txt"
    if not blacklist.exists():
        raise FileNotFoundError(f"metadata_root has no blacklist.txt: {metadata}")

    player_dirs = sorted(path for path in ttyrecs.iterdir() if path.is_dir())
    if not player_dirs:
        raise ValueError(f"ttyrec_root has no player directories: {ttyrecs}")

    staging.mkdir(parents=True, exist_ok=True)
    for path in [*xlogfiles, blacklist]:
        _symlink(path, staging / path.name)
    for path in player_dirs:
        _symlink(path, staging / path.name)

    return {
        "schema_version": "learn-nethack.altorg-staging.v1",
        "metadata_root": str(metadata),
        "ttyrec_root": str(ttyrecs),
        "staging_root": str(staging),
        "xlogfile_count": len(xlogfiles),
        "player_dir_count": len(player_dirs),
    }


def _symlink(source: Path, target: Path) -> None:
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.resolve() != source:
            raise FileExistsError(f"{target} already points somewhere else")
        return
    target.symlink_to(source, target_is_directory=source.is_dir())


def build_altorg_index(
    *,
    metadata_root: str | Path,
    ttyrec_root: str | Path,
    staging_root: str | Path,
    db_path: str | Path,
    dataset_name: str,
) -> dict[str, Any]:
    """Build an NLE ttyrecs.db for a local alt.org-style corpus."""
    target_db = Path(db_path)
    if target_db.exists():
        raise FileExistsError(f"index database already exists: {target_db}")
    target_db.parent.mkdir(parents=True, exist_ok=True)
    staging_report = prepare_altorg_staging_root(
        metadata_root=metadata_root,
        ttyrec_root=ttyrec_root,
        staging_root=staging_root,
    )

    try:
        from nle import dataset as nld
    except ImportError as exc:  # pragma: no cover - requires optional local-nle extra.
        raise RuntimeError("nle.dataset is required to index alt.org ttyrecs") from exc

    nld.db.create(filename=str(target_db))
    nld.populate_db.add_altorg_directory(
        str(Path(staging_root)),
        dataset_name,
        filename=str(target_db),
    )
    return {
        "schema_version": "learn-nethack.altorg-index.v1",
        "dataset_name": dataset_name,
        "db_path": str(target_db),
        "staging": staging_report,
    }
