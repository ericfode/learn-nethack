"""Resumable Modal volume upload helpers for large NLD player trees."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
from typing import IO, Iterable, Iterator, Protocol

from learn_nethack.nld_metadata import create_player_subset_db, read_ttyrec_players


class CommandRunner(Protocol):
    def __call__(
        self, command: list[str], *, timeout_seconds: int
    ) -> subprocess.CompletedProcess:
        """Run one upload command."""


@dataclass(frozen=True)
class ModalPlayerUpload:
    name: str
    source_path: str
    remote_parent: str

    def command(self, volume_name: str, *, force: bool = False) -> list[str]:
        command = [
            "modal",
            "volume",
            "put",
        ]
        if force:
            command.append("--force")
        command.extend([volume_name, self.source_path, self.remote_parent])
        return command


def completed_upload_names(progress_path: str | Path) -> set[str]:
    """Return player names with successful upload records."""
    path = Path(progress_path)
    if not path.exists():
        return set()
    names: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("status", "")).startswith("uploaded"):
                names.add(str(record["name"]))
    return names


def latest_upload_statuses(progress_path: str | Path) -> dict[str, str]:
    """Return the latest recorded upload status per player name."""
    path = Path(progress_path)
    if not path.exists():
        return {}
    statuses: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            statuses[str(record["name"])] = str(record.get("status", ""))
    return statuses


def resumable_skip_names(progress_path: str | Path) -> set[str]:
    """Return names to skip in broad resumable passes."""
    path = Path(progress_path)
    if not path.exists():
        return set()
    names: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            status = str(record.get("status", ""))
            if status.startswith("uploaded") or status in {"timeout", "deferred"}:
                names.add(str(record["name"]))
    return names


def plan_player_uploads(
    *,
    source_root: str | Path,
    remote_parent: str,
    completed_names: set[str] | None = None,
    only_names: set[str] | None = None,
    limit: int | None = None,
) -> list[ModalPlayerUpload]:
    """Plan one Modal upload command per top-level player directory."""
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"source_root does not exist: {root}")
    completed = completed_names or set()
    uploads: list[ModalPlayerUpload] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if only_names is not None and path.name not in only_names:
            continue
        if not path.is_dir() or path.name in completed:
            continue
        uploads.append(
            ModalPlayerUpload(
                name=path.name,
                source_path=str(path),
                remote_parent=remote_parent,
            )
        )
        if limit is not None and len(uploads) >= limit:
            break
    return uploads


def plan_indexed_player_uploads(
    *,
    source_root: str | Path,
    source_db: str | Path,
    remote_parent: str,
    completed_names: set[str] | None = None,
    limit: int | None = None,
) -> list[ModalPlayerUpload]:
    """Plan uploads for players that have ttyrecs in the NLD DB."""
    return plan_player_uploads(
        source_root=source_root,
        remote_parent=remote_parent,
        completed_names=completed_names,
        only_names=set(read_ttyrec_players(source_db)),
        limit=limit,
    )


def build_player_tar_shard(
    *,
    source_root: str | Path,
    player_names: Iterable[str],
    shard_path: str | Path,
) -> dict[str, object]:
    """Build a tar shard containing top-level player directories."""
    root = Path(source_root)
    if not root.exists():
        raise FileNotFoundError(f"source_root does not exist: {root}")
    target = Path(shard_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    players = sorted(set(player_names))
    with tarfile.open(target, "w", dereference=True) as archive:
        for player in players:
            player_path = root / player
            if not player_path.is_dir():
                raise FileNotFoundError(
                    f"player directory does not exist: {player_path}"
                )
            archive.add(player_path, arcname=player, recursive=True)
    return {
        "schema_version": "learn-nethack.nld-shard.v1",
        "shard_path": str(target),
        "source_root": str(root),
        "players": players,
        "player_count": len(players),
        "size_bytes": target.stat().st_size,
    }


def safe_extract_tar_shard(
    *,
    shard_path: str | Path,
    destination_root: str | Path,
) -> dict[str, object]:
    """Safely extract a player tar shard and reject path traversal or links."""
    shard = Path(shard_path)
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    players: set[str] = set()
    with tarfile.open(shard, "r") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_tar_member(member, destination_resolved)
            first_part = Path(member.name).parts[0]
            if first_part:
                players.add(first_part)
        archive.extractall(destination, members=members)
    return {
        "schema_version": "learn-nethack.nld-shard-extract-report.v1",
        "shard_path": str(shard),
        "destination_root": str(destination),
        "players": sorted(players),
        "player_count": len(players),
        "member_count": len(members),
    }


def append_shard_upload_records(
    *,
    progress_path: str | Path,
    player_names: Iterable[str],
    source_root: str | Path,
    remote_parent: str,
    shard_path: str | Path,
    status: str = "uploaded_shard",
) -> int:
    """Append per-player success rows after a shard has been extracted."""
    progress = Path(progress_path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source_root)
    names = sorted(set(player_names))
    with progress.open("a", encoding="utf-8") as handle:
        for name in names:
            _write_progress_record(
                handle,
                {
                    "name": name,
                    "source_path": str(source / name),
                    "remote_parent": remote_parent,
                    "status": status,
                    "returncode": 0,
                    "elapsed_seconds": 0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "shard_path": str(shard_path),
                },
            )
    return len(names)


def append_archive_shard_upload_records(
    *,
    progress_path: str | Path,
    player_names: Iterable[str],
    source_root: str | Path,
    remote_shard_path: str | Path,
    remote_subset_db_path: str | Path,
    subset_db_report: dict[str, object],
    status: str = "uploaded_archive_shard",
) -> int:
    """Append per-player success rows after tar and DB sidecar upload."""
    progress = Path(progress_path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    source = Path(source_root)
    names = sorted(set(player_names))
    selected_ttyrec_count = int(subset_db_report.get("selected_ttyrec_count", 0))
    selected_game_count = int(subset_db_report.get("selected_game_count", 0))
    with progress.open("a", encoding="utf-8") as handle:
        for name in names:
            _write_progress_record(
                handle,
                {
                    "name": name,
                    "source_path": str(source / name),
                    "status": status,
                    "returncode": 0,
                    "elapsed_seconds": 0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "remote_shard_path": str(remote_shard_path),
                    "remote_subset_db_path": str(remote_subset_db_path),
                    "selected_game_count": selected_game_count,
                    "selected_ttyrec_count": selected_ttyrec_count,
                },
            )
    return len(names)


def stage_player_archive_shard(
    *,
    source_root: str | Path,
    progress_path: str | Path,
    shard_path: str | Path,
    player_limit: int,
    source_db: str | Path,
    volume_name: str = "learn-nethack-datasets",
    remote_shard_parent: str = "/nld-shards/",
    remote_dataset_mount: str = "/datasets",
    subset_db_root: str = "/tmp/nld-shard",
    remote_subset_db_parent: str = "/nld-shard-dbs/",
    timeout_seconds: int = 60 * 60,
    command_runner: CommandRunner | None = None,
) -> dict[str, object]:
    """Build and upload a tar shard plus NLD sidecar DB without extraction."""
    if player_limit < 1:
        raise ValueError("player_limit must be positive")
    runner = _run_command if command_runner is None else command_runner
    completed = {
        name
        for name, status in latest_upload_statuses(progress_path).items()
        if status.startswith("uploaded_archive") or status == "deferred"
    }
    uploads = plan_indexed_player_uploads(
        source_root=source_root,
        source_db=source_db,
        remote_parent=remote_shard_parent,
        completed_names=completed,
        limit=player_limit,
    )
    player_names = [upload.name for upload in uploads]
    if not player_names:
        return {
            "schema_version": "learn-nethack.nld-archive-shard-stage-summary.v1",
            "status": "no_players",
            "player_count": 0,
            "shard_path": str(shard_path),
        }

    manifest = build_player_tar_shard(
        source_root=source_root,
        player_names=player_names,
        shard_path=shard_path,
    )
    shard = Path(shard_path)
    manifest_path = shard.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    subset_db_path = shard.with_suffix(".ttyrecs.db")
    subset_db_report = create_player_subset_db(
        source_db=source_db,
        target_db=subset_db_path,
        player_names=player_names,
        new_root=subset_db_root,
    )
    subset_report_path = shard.with_suffix(".db-report.json")
    subset_report_path.write_text(
        json.dumps(subset_db_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if int(subset_db_report["selected_ttyrec_count"]) == 0:
        return _archive_shard_stage_summary(
            status="empty_subset_db",
            manifest=manifest,
            manifest_path=manifest_path,
            player_names=player_names,
            upload_result=None,
            subset_db_path=subset_db_path,
            remote_shard_path=None,
            remote_subset_db_path=None,
            subset_db_report=subset_db_report,
            subset_db_upload_result=None,
            marked_count=0,
        )

    upload_result = runner(
        [
            "modal",
            "volume",
            "put",
            "--force",
            volume_name,
            str(shard),
            remote_shard_parent,
        ],
        timeout_seconds=timeout_seconds,
    )
    remote_shard_path = _remote_volume_mount_path(
        mount_path=remote_dataset_mount,
        volume_path=remote_shard_parent,
        filename=shard.name,
    )
    if upload_result.returncode != 0:
        return _archive_shard_stage_summary(
            status="upload_failed",
            manifest=manifest,
            manifest_path=manifest_path,
            player_names=player_names,
            upload_result=upload_result,
            subset_db_path=subset_db_path,
            remote_shard_path=None,
            remote_subset_db_path=None,
            subset_db_report=subset_db_report,
            subset_db_upload_result=None,
            marked_count=0,
        )

    subset_db_upload_result = runner(
        [
            "modal",
            "volume",
            "put",
            "--force",
            volume_name,
            str(subset_db_path),
            remote_subset_db_parent,
        ],
        timeout_seconds=timeout_seconds,
    )
    remote_subset_db_path = _remote_volume_mount_path(
        mount_path=remote_dataset_mount,
        volume_path=remote_subset_db_parent,
        filename=subset_db_path.name,
    )
    if subset_db_upload_result.returncode != 0:
        return _archive_shard_stage_summary(
            status="subset_db_upload_failed",
            manifest=manifest,
            manifest_path=manifest_path,
            player_names=player_names,
            upload_result=upload_result,
            subset_db_path=subset_db_path,
            remote_shard_path=remote_shard_path,
            remote_subset_db_path=None,
            subset_db_report=subset_db_report,
            subset_db_upload_result=subset_db_upload_result,
            marked_count=0,
        )

    marked_count = append_archive_shard_upload_records(
        progress_path=progress_path,
        player_names=player_names,
        source_root=source_root,
        remote_shard_path=remote_shard_path,
        remote_subset_db_path=remote_subset_db_path,
        subset_db_report=subset_db_report,
    )
    return _archive_shard_stage_summary(
        status="completed",
        manifest=manifest,
        manifest_path=manifest_path,
        player_names=player_names,
        upload_result=upload_result,
        subset_db_path=subset_db_path,
        remote_shard_path=remote_shard_path,
        remote_subset_db_path=remote_subset_db_path,
        subset_db_report=subset_db_report,
        subset_db_upload_result=subset_db_upload_result,
        marked_count=marked_count,
    )


def stage_player_tar_shard(
    *,
    source_root: str | Path,
    progress_path: str | Path,
    shard_path: str | Path,
    player_limit: int,
    volume_name: str = "learn-nethack-datasets",
    remote_parent: str = "/nld-nao-unzipped/",
    remote_shard_parent: str = "/nld-shards/",
    remote_dataset_mount: str = "/datasets",
    remote_extract_destination: str = "/datasets/nld-nao-unzipped",
    extract_entrypoint: str = "src/learn_nethack/modal_train.py::extract_nld_shard",
    extract_report: str | None = None,
    source_db: str | Path | None = None,
    subset_db_root: str = "/tmp/nld-shard",
    remote_subset_db_parent: str = "/nld-shard-dbs/",
    timeout_seconds: int = 60 * 60,
    command_runner: CommandRunner | None = None,
) -> dict[str, object]:
    """Build, upload, extract, and ledger one player tar shard."""
    if player_limit < 1:
        raise ValueError("player_limit must be positive")
    runner = _run_command if command_runner is None else command_runner
    completed = {
        name
        for name, status in latest_upload_statuses(progress_path).items()
        if status.startswith("uploaded") or status == "deferred"
    }
    uploads = plan_player_uploads(
        source_root=source_root,
        remote_parent=remote_parent,
        completed_names=completed,
        limit=player_limit,
    )
    player_names = [upload.name for upload in uploads]
    if not player_names:
        return {
            "schema_version": "learn-nethack.nld-shard-stage-summary.v1",
            "status": "no_players",
            "player_count": 0,
            "shard_path": str(shard_path),
        }

    manifest = build_player_tar_shard(
        source_root=source_root,
        player_names=player_names,
        shard_path=shard_path,
    )
    shard = Path(shard_path)
    manifest_path = shard.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    upload_command = [
        "modal",
        "volume",
        "put",
        "--force",
        volume_name,
        str(shard),
        remote_shard_parent,
    ]
    upload_result = runner(upload_command, timeout_seconds=timeout_seconds)
    if upload_result.returncode != 0:
        return _shard_stage_summary(
            status="upload_failed",
            manifest=manifest,
            manifest_path=manifest_path,
            player_names=player_names,
            upload_result=upload_result,
            extract_result=None,
            marked_count=0,
        )

    subset_db_path = None
    remote_subset_db_path = None
    subset_db_report = None
    subset_db_upload_result = None
    if source_db is not None:
        subset_db_path = shard.with_suffix(".ttyrecs.db")
        subset_db_report = create_player_subset_db(
            source_db=source_db,
            target_db=subset_db_path,
            player_names=player_names,
            new_root=subset_db_root,
        )
        subset_db_upload_result = runner(
            [
                "modal",
                "volume",
                "put",
                "--force",
                volume_name,
                str(subset_db_path),
                remote_subset_db_parent,
            ],
            timeout_seconds=timeout_seconds,
        )
        if subset_db_upload_result.returncode != 0:
            return _shard_stage_summary(
                status="subset_db_upload_failed",
                manifest=manifest,
                manifest_path=manifest_path,
                player_names=player_names,
                upload_result=upload_result,
                extract_result=None,
                marked_count=0,
                subset_db_path=subset_db_path,
                remote_subset_db_path=None,
                subset_db_report=subset_db_report,
                subset_db_upload_result=subset_db_upload_result,
            )
        remote_subset_db_path = _remote_volume_mount_path(
            mount_path=remote_dataset_mount,
            volume_path=remote_subset_db_parent,
            filename=subset_db_path.name,
        )

    remote_shard_path = _remote_volume_mount_path(
        mount_path=remote_dataset_mount,
        volume_path=remote_shard_parent,
        filename=shard.name,
    )
    report_path = extract_report or (
        f"/runs/nld-shard-extract/reports/{shard.stem}.json"
    )
    extract_command = [
        "modal",
        "run",
        extract_entrypoint,
        "--shard",
        remote_shard_path,
        "--destination",
        remote_extract_destination,
        "--report",
        report_path,
    ]
    extract_result = runner(extract_command, timeout_seconds=timeout_seconds)
    if extract_result.returncode != 0:
        return _shard_stage_summary(
            status="extract_failed",
            manifest=manifest,
            manifest_path=manifest_path,
            player_names=player_names,
            upload_result=upload_result,
            extract_result=extract_result,
            marked_count=0,
            subset_db_path=subset_db_path,
            remote_subset_db_path=remote_subset_db_path,
            subset_db_report=subset_db_report,
            subset_db_upload_result=subset_db_upload_result,
        )

    marked_count = append_shard_upload_records(
        progress_path=progress_path,
        player_names=player_names,
        source_root=source_root,
        remote_parent=remote_parent,
        shard_path=remote_shard_path,
    )
    return _shard_stage_summary(
        status="completed",
        manifest=manifest,
        manifest_path=manifest_path,
        player_names=player_names,
        upload_result=upload_result,
        extract_result=extract_result,
        marked_count=marked_count,
        subset_db_path=subset_db_path,
        remote_subset_db_path=remote_subset_db_path,
        subset_db_report=subset_db_report,
        subset_db_upload_result=subset_db_upload_result,
    )


def _remote_volume_mount_path(
    *, mount_path: str, volume_path: str, filename: str
) -> str:
    parent = volume_path.strip("/")
    if not parent:
        return f"{mount_path.rstrip('/')}/{filename}"
    return f"{mount_path.rstrip('/')}/{parent}/{filename}"


def _shard_stage_summary(
    *,
    status: str,
    manifest: dict[str, object],
    manifest_path: Path,
    player_names: list[str],
    upload_result: subprocess.CompletedProcess,
    extract_result: subprocess.CompletedProcess | None,
    marked_count: int,
    subset_db_path: Path | None = None,
    remote_subset_db_path: str | None = None,
    subset_db_report: dict[str, object] | None = None,
    subset_db_upload_result: subprocess.CompletedProcess | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "learn-nethack.nld-shard-stage-summary.v1",
        "status": status,
        "shard_path": manifest["shard_path"],
        "manifest_path": str(manifest_path),
        "size_bytes": manifest["size_bytes"],
        "player_count": len(player_names),
        "first_player": player_names[0] if player_names else None,
        "last_player": player_names[-1] if player_names else None,
        "marked_count": marked_count,
        "upload_returncode": upload_result.returncode,
        "extract_returncode": None
        if extract_result is None
        else extract_result.returncode,
        "subset_db_path": None if subset_db_path is None else str(subset_db_path),
        "remote_subset_db_path": remote_subset_db_path,
        "subset_db_report": subset_db_report,
        "subset_db_upload_returncode": None
        if subset_db_upload_result is None
        else subset_db_upload_result.returncode,
        "upload_stdout_tail": upload_result.stdout[-2000:],
        "upload_stderr_tail": upload_result.stderr[-2000:],
        "extract_stdout_tail": ""
        if extract_result is None
        else extract_result.stdout[-2000:],
        "extract_stderr_tail": ""
        if extract_result is None
        else extract_result.stderr[-2000:],
    }


def _archive_shard_stage_summary(
    *,
    status: str,
    manifest: dict[str, object],
    manifest_path: Path,
    player_names: list[str],
    upload_result: subprocess.CompletedProcess | None,
    subset_db_path: Path,
    remote_shard_path: str | None,
    remote_subset_db_path: str | None,
    subset_db_report: dict[str, object],
    subset_db_upload_result: subprocess.CompletedProcess | None,
    marked_count: int,
) -> dict[str, object]:
    return {
        "schema_version": "learn-nethack.nld-archive-shard-stage-summary.v1",
        "status": status,
        "shard_path": manifest["shard_path"],
        "manifest_path": str(manifest_path),
        "size_bytes": manifest["size_bytes"],
        "player_count": len(player_names),
        "first_player": player_names[0] if player_names else None,
        "last_player": player_names[-1] if player_names else None,
        "marked_count": marked_count,
        "remote_shard_path": remote_shard_path,
        "subset_db_path": str(subset_db_path),
        "remote_subset_db_path": remote_subset_db_path,
        "subset_db_report": subset_db_report,
        "upload_returncode": None
        if upload_result is None
        else upload_result.returncode,
        "subset_db_upload_returncode": None
        if subset_db_upload_result is None
        else subset_db_upload_result.returncode,
        "upload_stdout_tail": ""
        if upload_result is None
        else upload_result.stdout[-2000:],
        "upload_stderr_tail": ""
        if upload_result is None
        else upload_result.stderr[-2000:],
        "subset_db_upload_stdout_tail": ""
        if subset_db_upload_result is None
        else subset_db_upload_result.stdout[-2000:],
        "subset_db_upload_stderr_tail": ""
        if subset_db_upload_result is None
        else subset_db_upload_result.stderr[-2000:],
    }


def _validate_tar_member(member: tarfile.TarInfo, destination: Path) -> None:
    member_path = Path(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"unsafe tar member path: {member.name}")
    if member.issym() or member.islnk():
        raise ValueError(f"unsafe tar member link: {member.name}")
    target = (destination / member.name).resolve()
    if not target.is_relative_to(destination):
        raise ValueError(f"unsafe tar member target: {member.name}")


def run_player_uploads(
    *,
    uploads: Iterable[ModalPlayerUpload],
    volume_name: str,
    progress_path: str | Path,
    force: bool = True,
    timeout_seconds: int = 300,
    jobs: int = 1,
    command_runner: CommandRunner | None = None,
) -> dict[str, int | str]:
    """Execute player uploads and append resumable JSONL progress records."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    runner = _run_command if command_runner is None else command_runner
    progress = Path(progress_path)
    progress.parent.mkdir(parents=True, exist_ok=True)
    uploaded = 0
    failed = 0
    with progress.open("a", encoding="utf-8") as handle:
        if jobs == 1:
            for upload in uploads:
                record = _run_one_player_upload(
                    upload=upload,
                    volume_name=volume_name,
                    force=force,
                    timeout_seconds=timeout_seconds,
                    command_runner=runner,
                )
                status = str(record["status"])
                if status == "uploaded":
                    uploaded += 1
                else:
                    failed += 1
                _write_progress_record(handle, record)
                if status == "failed":
                    break
        else:
            uploads_iter = iter(uploads)
            stop_scheduling = False
            pending: dict[Future[dict[str, object]], ModalPlayerUpload] = {}
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                pending.update(
                    _submit_uploads(
                        executor=executor,
                        uploads_iter=uploads_iter,
                        volume_name=volume_name,
                        force=force,
                        timeout_seconds=timeout_seconds,
                        command_runner=runner,
                        limit=jobs,
                    )
                )
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        record = future.result()
                        status = str(record["status"])
                        if status == "uploaded":
                            uploaded += 1
                        else:
                            failed += 1
                        _write_progress_record(handle, record)
                        if status == "failed":
                            stop_scheduling = True
                    if not stop_scheduling:
                        pending.update(
                            _submit_uploads(
                                executor=executor,
                                uploads_iter=uploads_iter,
                                volume_name=volume_name,
                                force=force,
                                timeout_seconds=timeout_seconds,
                                command_runner=runner,
                                limit=jobs - len(pending),
                            )
                        )
    return {
        "schema_version": "learn-nethack.modal-player-upload-summary.v1",
        "progress_path": str(progress),
        "uploaded": uploaded,
        "failed": failed,
    }


def _submit_uploads(
    *,
    executor: ThreadPoolExecutor,
    uploads_iter: Iterator[ModalPlayerUpload],
    volume_name: str,
    force: bool,
    timeout_seconds: int,
    command_runner: CommandRunner,
    limit: int,
) -> dict[Future[dict[str, object]], ModalPlayerUpload]:
    pending: dict[Future[dict[str, object]], ModalPlayerUpload] = {}
    for _ in range(limit):
        try:
            upload = next(uploads_iter)
        except StopIteration:
            break
        future = executor.submit(
            _run_one_player_upload,
            upload=upload,
            volume_name=volume_name,
            force=force,
            timeout_seconds=timeout_seconds,
            command_runner=command_runner,
        )
        pending[future] = upload
    return pending


def _run_one_player_upload(
    *,
    upload: ModalPlayerUpload,
    volume_name: str,
    force: bool,
    timeout_seconds: int,
    command_runner: CommandRunner,
) -> dict[str, object]:
    started_at = time.time()
    command = upload.command(volume_name, force=force)
    try:
        result = command_runner(command, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        returncode = None
        stdout = _text_from_timeout_field(exc.stdout)
        stderr = _text_from_timeout_field(exc.stderr)
    else:
        status = "uploaded" if result.returncode == 0 else "failed"
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    return {
        "name": upload.name,
        "source_path": upload.source_path,
        "remote_parent": upload.remote_parent,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def _text_from_timeout_field(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _write_progress_record(handle: IO[str], record: dict[str, object]) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def _run_command(
    command: list[str], *, timeout_seconds: int
) -> subprocess.CompletedProcess:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=command,
            timeout=timeout_seconds,
            output=stdout,
            stderr=stderr,
        ) from exc
    except BaseException:
        _terminate_process_group(process.pid)
        raise
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except (PermissionError, ProcessLookupError):
        return
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            os.killpg(pid, 0)
        except (PermissionError, ProcessLookupError):
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        return


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload NLD player directories to a Modal volume one at a time."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--remote-parent", default="/nld-nao-unzipped/")
    parser.add_argument("--volume-name", default="learn-nethack-datasets")
    parser.add_argument(
        "--progress",
        default="artifacts/modal-upload-nld-nao-players.jsonl",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--only-name", action="append", dest="only_names")
    parser.add_argument("--no-force", action="store_true")
    args = parser.parse_args()

    only_names = set(args.only_names) if args.only_names else None
    completed = set() if only_names is not None else resumable_skip_names(args.progress)
    uploads = plan_player_uploads(
        source_root=args.source_root,
        remote_parent=args.remote_parent,
        completed_names=completed,
        only_names=only_names,
        limit=args.limit,
    )
    summary = run_player_uploads(
        uploads=uploads,
        volume_name=args.volume_name,
        progress_path=args.progress,
        force=not args.no_force,
        timeout_seconds=args.timeout_seconds,
        jobs=args.jobs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
