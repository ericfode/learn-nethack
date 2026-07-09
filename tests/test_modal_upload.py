from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sqlite3
import tarfile
import threading
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from learn_nethack import modal_upload
from learn_nethack.modal_upload import (
    ModalPlayerUpload,
    append_archive_shard_upload_records,
    append_shard_upload_records,
    build_player_tar_shard,
    completed_upload_names,
    latest_upload_statuses,
    plan_indexed_player_uploads,
    plan_player_uploads,
    run_player_uploads,
    safe_extract_tar_shard,
    stage_player_archive_shard,
    stage_player_tar_shard,
    resumable_skip_names,
)


class ModalUploadTests(unittest.TestCase):
    def test_plan_player_uploads_sorts_dirs_and_skips_completed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("zed", "ana", "bob"):
                (root / name).mkdir()
            (root / "README.txt").write_text("not a player", encoding="utf-8")

            uploads = plan_player_uploads(
                source_root=root,
                remote_parent="/nld-nao-unzipped/",
                completed_names={"ana"},
                limit=2,
            )

        self.assertEqual(
            uploads,
            [
                ModalPlayerUpload(
                    name="bob",
                    source_path=str(root / "bob"),
                    remote_parent="/nld-nao-unzipped/",
                ),
                ModalPlayerUpload(
                    name="zed",
                    source_path=str(root / "zed"),
                    remote_parent="/nld-nao-unzipped/",
                ),
            ],
        )
        self.assertEqual(
            uploads[0].command("learn-nethack-datasets", force=True),
            [
                "modal",
                "volume",
                "put",
                "--force",
                "learn-nethack-datasets",
                str(root / "bob"),
                "/nld-nao-unzipped/",
            ],
        )

    def test_plan_player_uploads_can_target_named_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("zed", "ana", "bob"):
                (root / name).mkdir()

            uploads = plan_player_uploads(
                source_root=root,
                remote_parent="/nld-nao-unzipped/",
                only_names={"ana", "missing"},
            )

        self.assertEqual(
            uploads,
            [
                ModalPlayerUpload(
                    name="ana",
                    source_path=str(root / "ana"),
                    remote_parent="/nld-nao-unzipped/",
                )
            ],
        )

    def test_plan_indexed_player_uploads_uses_db_players_only(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            for name in ("ana", "bob", "ghost", "zed"):
                (source / name).mkdir()
            db_path = root / "ttyrecs.db"
            with sqlite3.connect(db_path) as conn:
                conn.executescript(
                    """
                    create table ttyrecs (path text, part integer, size integer, mtime real, gameid integer);
                    insert into ttyrecs values ('bob/b.ttyrec.bz2', 0, 1, 0.0, 1);
                    insert into ttyrecs values ('zed/z.ttyrec.bz2', 0, 1, 0.0, 2);
                    """
                )

            uploads = plan_indexed_player_uploads(
                source_root=source,
                source_db=db_path,
                remote_parent="/nld-shards/",
                completed_names={"bob"},
                limit=2,
            )

        self.assertEqual([upload.name for upload in uploads], ["zed"])

    def test_completed_upload_names_reads_successful_jsonl_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.jsonl"
            progress.write_text(
                "\n".join(
                    [
                        json.dumps({"name": "ana", "status": "uploaded"}),
                        json.dumps({"name": "cee", "status": "uploaded_verified"}),
                        json.dumps({"name": "bob", "status": "failed"}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            names = completed_upload_names(progress)

        self.assertEqual(names, {"ana", "cee"})

    def test_latest_upload_statuses_uses_last_record_per_name(self) -> None:
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.jsonl"
            progress.write_text(
                "\n".join(
                    [
                        json.dumps({"name": "ana", "status": "timeout"}),
                        json.dumps({"name": "bob", "status": "uploaded"}),
                        json.dumps({"name": "ana", "status": "uploaded"}),
                    ]
                ),
                encoding="utf-8",
            )

            statuses = latest_upload_statuses(progress)

        self.assertEqual(statuses, {"ana": "uploaded", "bob": "uploaded"})

    def test_resumable_skip_names_includes_deferred_timeouts(self) -> None:
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.jsonl"
            progress.write_text(
                "\n".join(
                    [
                        json.dumps({"name": "ana", "status": "uploaded"}),
                        json.dumps({"name": "tim", "status": "timeout"}),
                        json.dumps({"name": "bob", "status": "failed"}),
                    ]
                ),
                encoding="utf-8",
            )

            names = resumable_skip_names(progress)

        self.assertEqual(names, {"ana", "tim"})

    def test_run_player_uploads_supports_parallel_jobs(self) -> None:
        uploads = [
            ModalPlayerUpload(
                name=f"player-{index}",
                source_path=f"/src/{index}",
                remote_parent="/dst/",
            )
            for index in range(4)
        ]
        active = 0
        max_active = 0
        lock = threading.Lock()

        def runner(
            command: list[str], *, timeout_seconds: int
        ) -> subprocess.CompletedProcess:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="uploaded",
                stderr="",
            )

        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.jsonl"

            summary = run_player_uploads(
                uploads=uploads,
                volume_name="learn-nethack-datasets",
                progress_path=progress,
                jobs=2,
                command_runner=runner,
            )

            rows = [
                json.loads(line)
                for line in progress.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(summary["uploaded"], 4)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(max_active, 2)
        self.assertEqual({row["status"] for row in rows}, {"uploaded"})
        self.assertEqual(
            {row["name"] for row in rows}, {upload.name for upload in uploads}
        )

    def test_run_player_uploads_rejects_non_positive_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "jobs must be positive"):
                run_player_uploads(
                    uploads=[],
                    volume_name="learn-nethack-datasets",
                    progress_path=Path(tmp) / "progress.jsonl",
                    jobs=0,
                )

    def test_terminate_process_group_treats_permission_denied_probe_as_gone(
        self,
    ) -> None:
        with patch.object(
            modal_upload.os,
            "killpg",
            side_effect=[None, PermissionError()],
        ):
            modal_upload._terminate_process_group(123)

    def test_run_command_terminates_process_group_on_keyboard_interrupt(
        self,
    ) -> None:
        class FakeProcess:
            pid = 123

            def communicate(
                self, timeout: int
            ) -> tuple[str, str]:  # pragma: no cover - never returns.
                raise KeyboardInterrupt

        with (
            patch.object(modal_upload.subprocess, "Popen", return_value=FakeProcess()),
            patch.object(modal_upload, "_terminate_process_group") as terminate,
        ):
            with self.assertRaises(KeyboardInterrupt):
                modal_upload._run_command(["modal"], timeout_seconds=1)

        terminate.assert_called_once_with(123)

    def test_build_and_extract_player_tar_shard(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            (root / "bob").mkdir()
            (root / "bob" / "b.ttyrec.bz2").write_text("bob", encoding="utf-8")
            (root / "ana").mkdir()
            (root / "ana" / "nested").mkdir()
            (root / "ana" / "nested" / "a.ttyrec.bz2").write_text(
                "ana",
                encoding="utf-8",
            )
            shard_path = Path(tmp) / "shards" / "nld-nao-000001.tar"

            manifest = build_player_tar_shard(
                source_root=root,
                player_names=["bob", "ana"],
                shard_path=shard_path,
            )
            destination = Path(tmp) / "extracted"
            extract_report = safe_extract_tar_shard(
                shard_path=shard_path,
                destination_root=destination,
            )

            self.assertEqual(manifest["schema_version"], "learn-nethack.nld-shard.v1")
            self.assertEqual(manifest["players"], ["ana", "bob"])
            self.assertEqual(manifest["player_count"], 2)
            self.assertEqual(extract_report["player_count"], 2)
            self.assertEqual(
                (destination / "ana" / "nested" / "a.ttyrec.bz2").read_text(
                    encoding="utf-8"
                ),
                "ana",
            )
            self.assertEqual(
                (destination / "bob" / "b.ttyrec.bz2").read_text(encoding="utf-8"),
                "bob",
            )

    def test_build_player_tar_shard_dereferences_player_symlink_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_source = root / "real"
            real_source.mkdir()
            (real_source / "ana").mkdir()
            (real_source / "ana" / "a.ttyrec.bz2").write_text(
                "ana",
                encoding="utf-8",
            )
            staging = root / "staging"
            staging.mkdir()
            (staging / "ana").symlink_to(real_source / "ana", target_is_directory=True)
            shard_path = root / "shards" / "symlink-player.tar"

            build_player_tar_shard(
                source_root=staging,
                player_names=["ana"],
                shard_path=shard_path,
            )
            destination = root / "extracted"
            report = safe_extract_tar_shard(
                shard_path=shard_path,
                destination_root=destination,
            )

            self.assertEqual(report["player_count"], 1)
            self.assertFalse((destination / "ana").is_symlink())
            self.assertEqual(
                (destination / "ana" / "a.ttyrec.bz2").read_text(encoding="utf-8"),
                "ana",
            )

    def test_safe_extract_tar_shard_rejects_path_traversal(self) -> None:
        with TemporaryDirectory() as tmp:
            shard_path = Path(tmp) / "unsafe.tar"
            with tarfile.open(shard_path, "w") as archive:
                info = tarfile.TarInfo("../escape.txt")
                payload = b"bad"
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

            with self.assertRaisesRegex(ValueError, "unsafe tar member"):
                safe_extract_tar_shard(
                    shard_path=shard_path,
                    destination_root=Path(tmp) / "dest",
                )

        self.assertFalse((Path(tmp) / "escape.txt").exists())

    def test_append_shard_upload_records_updates_latest_statuses(self) -> None:
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "progress.jsonl"

            append_shard_upload_records(
                progress_path=progress,
                player_names=["ana", "bob"],
                source_root="/source",
                remote_parent="/nld-nao-unzipped/",
                shard_path="/datasets/nld-shards/shard-1.tar",
            )

            rows = [
                json.loads(line)
                for line in progress.read_text(encoding="utf-8").splitlines()
            ]
            statuses = latest_upload_statuses(progress)

        self.assertEqual(statuses, {"ana": "uploaded_shard", "bob": "uploaded_shard"})
        self.assertEqual(
            {row["shard_path"] for row in rows}, {"/datasets/nld-shards/shard-1.tar"}
        )

    def test_append_archive_shard_upload_records_writes_tar_and_db_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            progress = Path(tmp) / "archive-progress.jsonl"

            append_archive_shard_upload_records(
                progress_path=progress,
                player_names=["ana", "bob"],
                source_root="/source",
                remote_shard_path="/datasets/nld-shards/shard-1.tar",
                remote_subset_db_path="/datasets/nld-shard-dbs/shard-1.ttyrecs.db",
                subset_db_report={"selected_ttyrec_count": 4},
            )

            rows = [
                json.loads(line)
                for line in progress.read_text(encoding="utf-8").splitlines()
            ]
            statuses = latest_upload_statuses(progress)

        self.assertEqual(
            statuses,
            {"ana": "uploaded_archive_shard", "bob": "uploaded_archive_shard"},
        )
        self.assertEqual(
            {row["remote_subset_db_path"] for row in rows},
            {"/datasets/nld-shard-dbs/shard-1.ttyrecs.db"},
        )
        self.assertEqual({row["selected_ttyrec_count"] for row in rows}, {4})

    def test_stage_player_tar_shard_uploads_extracts_then_marks_ledger(self) -> None:
        commands: list[list[str]] = []

        def runner(
            command: list[str], *, timeout_seconds: int
        ) -> subprocess.CompletedProcess:
            commands.append(command)
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="ok",
                stderr="",
            )

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            for name in ("ana", "bob"):
                (source / name).mkdir()
                (source / name / f"{name}.ttyrec.bz2").write_text(
                    name,
                    encoding="utf-8",
                )
            progress = Path(tmp) / "progress.jsonl"
            shard_path = Path(tmp) / "shards" / "nld-nao-shard-test.tar"

            summary = stage_player_tar_shard(
                source_root=source,
                progress_path=progress,
                shard_path=shard_path,
                player_limit=2,
                command_runner=runner,
            )

            statuses = latest_upload_statuses(progress)
            manifest_path = Path(str(summary["manifest_path"]))
            self.assertTrue(manifest_path.exists())

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["player_count"], 2)
        self.assertEqual(statuses, {"ana": "uploaded_shard", "bob": "uploaded_shard"})
        self.assertEqual(
            commands,
            [
                [
                    "modal",
                    "volume",
                    "put",
                    "--force",
                    "learn-nethack-datasets",
                    str(shard_path),
                    "/nld-shards/",
                ],
                [
                    "modal",
                    "run",
                    "src/learn_nethack/modal_train.py::extract_nld_shard",
                    "--shard",
                    "/datasets/nld-shards/nld-nao-shard-test.tar",
                    "--destination",
                    "/datasets/nld-nao-unzipped",
                    "--report",
                    "/runs/nld-shard-extract/reports/nld-nao-shard-test.json",
                ],
            ],
        )

    def test_stage_player_tar_shard_can_upload_subset_db_sidecar(self) -> None:
        commands: list[list[str]] = []

        def runner(
            command: list[str], *, timeout_seconds: int
        ) -> subprocess.CompletedProcess:
            commands.append(command)
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "ana").mkdir()
            (source / "ana" / "a.ttyrec.bz2").write_text("ana", encoding="utf-8")
            source_db = Path(tmp) / "ttyrecs.db"
            with sqlite3.connect(source_db) as conn:
                conn.executescript(
                    """
                    create table meta (ctime real, mtime real);
                    create table roots (dataset_name text primary key, root text, ttyrec_version integer);
                    create table games (gameid integer primary key, role text, race text, align text, death text, points integer, turns integer);
                    create table datasets (gameid integer, dataset_name text, primary key(dataset_name, gameid));
                    create table ttyrecs (path text, part integer, size integer, mtime real, gameid integer, primary key(gameid, part, path));
                    insert into meta values (1.0, 2.0);
                    insert into roots values ('nld-nao', '/old', 1);
                    insert into games values (1, 'Val', 'Hum', 'Law', 'quit', 1, 2);
                    insert into datasets values (1, 'nld-nao');
                    insert into ttyrecs values ('ana/a.ttyrec.bz2', 0, 3, 0.0, 1);
                    """
                )
            progress = Path(tmp) / "progress.jsonl"
            shard_path = Path(tmp) / "shards" / "nld-nao-shard-test.tar"

            summary = stage_player_tar_shard(
                source_root=source,
                progress_path=progress,
                shard_path=shard_path,
                player_limit=1,
                source_db=source_db,
                subset_db_root="/tmp/nld-shard",
                command_runner=runner,
            )

            subset_db_path = Path(str(summary["subset_db_path"]))
            self.assertTrue(subset_db_path.exists())

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(
            summary["remote_subset_db_path"],
            "/datasets/nld-shard-dbs/nld-nao-shard-test.ttyrecs.db",
        )
        self.assertEqual(
            commands[1][:6],
            [
                "modal",
                "volume",
                "put",
                "--force",
                "learn-nethack-datasets",
                str(subset_db_path),
            ],
        )
        self.assertEqual(commands[1][6], "/nld-shard-dbs/")

    def test_stage_player_archive_shard_uploads_tar_and_db_without_extract(
        self,
    ) -> None:
        commands: list[list[str]] = []

        def runner(
            command: list[str], *, timeout_seconds: int
        ) -> subprocess.CompletedProcess:
            commands.append(command)
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="ok", stderr=""
            )

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            for name in ("ana", "bob", "ghost"):
                (source / name).mkdir()
                (source / name / f"{name}.ttyrec.bz2").write_text(
                    name,
                    encoding="utf-8",
                )
            source_db = Path(tmp) / "ttyrecs.db"
            with sqlite3.connect(source_db) as conn:
                conn.executescript(
                    """
                    create table meta (ctime real, mtime real);
                    create table roots (dataset_name text primary key, root text, ttyrec_version integer);
                    create table games (gameid integer primary key, role text, race text, align text, death text, points integer, turns integer);
                    create table datasets (gameid integer, dataset_name text, primary key(dataset_name, gameid));
                    create table ttyrecs (path text, part integer, size integer, mtime real, gameid integer, primary key(gameid, part, path));
                    insert into meta values (1.0, 2.0);
                    insert into roots values ('nld-nao', '/old', 1);
                    insert into games values (1, 'Val', 'Hum', 'Law', 'quit', 1, 2);
                    insert into games values (2, 'Val', 'Hum', 'Law', 'quit', 1, 2);
                    insert into datasets values (1, 'nld-nao');
                    insert into datasets values (2, 'nld-nao');
                    insert into ttyrecs values ('ana/a.ttyrec.bz2', 0, 3, 0.0, 1);
                    insert into ttyrecs values ('bob/b.ttyrec.bz2', 0, 3, 0.0, 2);
                    """
                )
            progress = Path(tmp) / "archive-progress.jsonl"
            shard_path = Path(tmp) / "shards" / "nld-nao-shard-test.tar"

            summary = stage_player_archive_shard(
                source_root=source,
                progress_path=progress,
                shard_path=shard_path,
                player_limit=2,
                source_db=source_db,
                command_runner=runner,
            )

            statuses = latest_upload_statuses(progress)

        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["player_count"], 2)
        self.assertEqual(summary["marked_count"], 2)
        self.assertEqual(
            statuses, {"ana": "uploaded_archive_shard", "bob": "uploaded_archive_shard"}
        )
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            commands[0][:6],
            [
                "modal",
                "volume",
                "put",
                "--force",
                "learn-nethack-datasets",
                str(shard_path),
            ],
        )
        self.assertEqual(commands[0][6], "/nld-shards/")
        self.assertEqual(
            commands[1][0:5],
            ["modal", "volume", "put", "--force", "learn-nethack-datasets"],
        )
        self.assertNotIn(
            "src/learn_nethack/modal_train.py::extract_nld_shard",
            " ".join(" ".join(command) for command in commands),
        )

    def test_stage_player_tar_shard_does_not_mark_ledger_when_extract_fails(
        self,
    ) -> None:
        calls = 0

        def runner(
            command: list[str], *, timeout_seconds: int
        ) -> subprocess.CompletedProcess:
            nonlocal calls
            calls += 1
            return subprocess.CompletedProcess(
                args=command,
                returncode=0 if calls == 1 else 2,
                stdout="",
                stderr="extract failed",
            )

        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "ana").mkdir()
            (source / "ana" / "a.ttyrec.bz2").write_text("ana", encoding="utf-8")
            progress = Path(tmp) / "progress.jsonl"

            summary = stage_player_tar_shard(
                source_root=source,
                progress_path=progress,
                shard_path=Path(tmp) / "shard.tar",
                player_limit=1,
                command_runner=runner,
            )

            statuses = latest_upload_statuses(progress)

        self.assertEqual(summary["status"], "extract_failed")
        self.assertEqual(statuses, {})


if __name__ == "__main__":
    unittest.main()
