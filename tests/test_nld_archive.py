from __future__ import annotations

import sqlite3
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from learn_nethack.nld_archive import (
    ArchiveShard,
    iter_archive_dataset_batches,
    iter_nld_archive_shard_batches,
    plan_archive_dataset,
    read_archive_manifest,
)


def _make_fixture_db(
    path: Path,
    *,
    root: str = "/datasets/stale-root",
    dataset_name: str = "fixture-nld",
    gameid: int = 1,
    player: str = "ana",
) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "create table roots (dataset_name text primary key, root text, ttyrec_version integer)"
        )
        conn.execute(
            "create table games (gameid integer primary key, role text, race text, align text, death text, points integer, turns integer)"
        )
        conn.execute(
            "create table ttyrecs (path text, part integer, size integer, mtime real, gameid integer)"
        )
        conn.execute("insert into roots values (?, ?, 3)", (dataset_name, root))
        conn.execute(
            "insert into games values (?, 'Sam', 'Hum', 'Law', 'quit', 10, 20)",
            (gameid,),
        )
        conn.execute(
            "insert into ttyrecs values (?, 0, 100, 0.0, ?)",
            (f"{player}/a.ttyrec.bz2", gameid),
        )


def _make_player_tar(path: Path, *, player: str = "ana") -> None:
    source = path.parent / "source"
    ttyrec = source / player / "a.ttyrec.bz2"
    ttyrec.parent.mkdir(parents=True)
    ttyrec.write_text("ttyrec fixture", encoding="utf-8")
    with tarfile.open(path, "w") as archive:
        archive.add(ttyrec.parent, arcname=player, recursive=True)


class NldArchiveTests(unittest.TestCase):
    def test_read_archive_manifest_requires_explicit_shard_and_db_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard.tar"
            db = root / "ttyrecs.db"
            shard.write_text("not a real tar", encoding="utf-8")
            _make_fixture_db(db)
            manifest = root / "archive.jsonl"
            manifest.write_text(
                '{"shard_tar": "shard.tar", "shard_db": "ttyrecs.db"}\n',
                encoding="utf-8",
            )

            shards = read_archive_manifest(manifest)

        self.assertEqual(
            shards,
            [ArchiveShard(shard_tar=str(shard), shard_db=str(db))],
        )

    def test_iter_archive_shard_batches_rewrites_db_to_ephemeral_extract_root(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "shard.tar"
            db = root / "ttyrecs.db"
            _make_player_tar(shard)
            _make_fixture_db(db)
            observed: dict[str, str] = {}

            def fake_batch_iterator(**kwargs: Any):
                observed["dbfilename"] = kwargs["dbfilename"]
                observed["dataset_name"] = kwargs["dataset_name"]
                observed["gameids"] = repr(kwargs["gameids"])
                with sqlite3.connect(kwargs["dbfilename"]) as conn:
                    staged_root = conn.execute("select root from roots").fetchone()[0]
                observed["staged_root"] = staged_root
                observed["ttyrec_text"] = (
                    Path(staged_root) / "ana" / "a.ttyrec.bz2"
                ).read_text(encoding="utf-8")
                yield {
                    "gameids": [1],
                    "keypresses": [107],
                    "tty_chars": [[[64, 46]]],
                }

            batches = list(
                iter_nld_archive_shard_batches(
                    ArchiveShard(shard_tar=str(shard), shard_db=str(db)),
                    dataset_name="fixture-nld",
                    batch_size=4,
                    seq_length=8,
                    gameids=[1],
                    batch_iterator=fake_batch_iterator,
                )
            )
            staged_root = observed["staged_root"]

        self.assertEqual(len(batches), 1)
        self.assertEqual(observed["dataset_name"], "fixture-nld")
        self.assertEqual(observed["gameids"], "[1]")
        self.assertEqual(observed["ttyrec_text"], "ttyrec fixture")
        self.assertFalse(Path(staged_root).exists())

    def test_plan_archive_dataset_merges_shard_gameids_and_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_one = root / "one.db"
            db_two = root / "two.db"
            _make_fixture_db(db_one, gameid=1, player="ana")
            _make_fixture_db(db_two, gameid=2, player="bob")
            manifest = root / "archive.jsonl"
            manifest.write_text(
                '{"shard_tar": "one.tar", "shard_db": "one.db"}\n'
                '{"shard_tar": "two.tar", "shard_db": "two.db"}\n',
                encoding="utf-8",
            )

            plan = plan_archive_dataset(manifest)

        self.assertEqual(plan.dataset_name, "fixture-nld")
        self.assertEqual(plan.gameids, (1, 2))
        self.assertEqual(plan.gameids_for(plan.shards[0]), (1,))
        self.assertEqual(plan.gameids_for(plan.shards[1]), (2,))
        self.assertEqual(plan.game_metadata_by_id[2]["role"], "Sam")

    def test_plan_archive_dataset_rejects_mixed_dataset_names(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture_db(root / "one.db", dataset_name="fixture-a")
            _make_fixture_db(root / "two.db", dataset_name="fixture-b", gameid=2)
            manifest = root / "archive.jsonl"
            manifest.write_text(
                '{"shard_tar": "one.tar", "shard_db": "one.db"}\n'
                '{"shard_tar": "two.tar", "shard_db": "two.db"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dataset_name"):
                plan_archive_dataset(manifest)

    def test_plan_archive_dataset_deduplicates_global_gameids_across_shards(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture_db(root / "one.db", gameid=1, player="ana")
            _make_fixture_db(root / "two.db", gameid=1, player="bob")
            manifest = root / "archive.jsonl"
            manifest.write_text(
                '{"shard_tar": "one.tar", "shard_db": "one.db"}\n'
                '{"shard_tar": "two.tar", "shard_db": "two.db"}\n',
                encoding="utf-8",
            )

            plan = plan_archive_dataset(manifest)

        self.assertEqual(plan.gameids, (1,))
        self.assertEqual(plan.gameids_for(plan.shards[0]), (1,))
        self.assertEqual(plan.gameids_for(plan.shards[1]), (1,))

    def test_iter_archive_dataset_batches_routes_selected_gameids_per_shard(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_one = root / "one.tar"
            shard_two = root / "two.tar"
            db_one = root / "one.db"
            db_two = root / "two.db"
            _make_player_tar(shard_one, player="ana")
            _make_player_tar(shard_two, player="bob")
            _make_fixture_db(db_one, gameid=1, player="ana")
            _make_fixture_db(db_two, gameid=2, player="bob")
            manifest = root / "archive.jsonl"
            manifest.write_text(
                '{"shard_tar": "one.tar", "shard_db": "one.db"}\n'
                '{"shard_tar": "two.tar", "shard_db": "two.db"}\n',
                encoding="utf-8",
            )
            plan = plan_archive_dataset(manifest)
            routed_gameids: list[list[int]] = []

            def fake_batch_iterator(**kwargs: Any):
                routed_gameids.append(kwargs["gameids"])
                yield {
                    "gameids": kwargs["gameids"],
                    "keypresses": [107],
                    "tty_chars": [[[64, 46]]],
                }

            batches = list(
                iter_archive_dataset_batches(
                    plan,
                    selected_gameids=[2],
                    batch_size=4,
                    seq_length=8,
                    batch_iterator=fake_batch_iterator,
                )
            )

        self.assertEqual(len(batches), 1)
        self.assertEqual(routed_gameids, [[2]])

    def test_iter_archive_dataset_batches_can_select_shard_indices(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_one = root / "one.tar"
            shard_two = root / "two.tar"
            db_one = root / "one.db"
            db_two = root / "two.db"
            _make_player_tar(shard_one, player="ana")
            _make_player_tar(shard_two, player="bob")
            _make_fixture_db(db_one, gameid=1, player="ana")
            _make_fixture_db(db_two, gameid=2, player="bob")
            manifest = root / "archive.jsonl"
            manifest.write_text(
                '{"shard_tar": "one.tar", "shard_db": "one.db"}\n'
                '{"shard_tar": "two.tar", "shard_db": "two.db"}\n',
                encoding="utf-8",
            )
            plan = plan_archive_dataset(manifest)
            routed_gameids: list[list[int]] = []

            def fake_batch_iterator(**kwargs: Any):
                routed_gameids.append(kwargs["gameids"])
                yield {
                    "gameids": kwargs["gameids"],
                    "keypresses": [107],
                    "tty_chars": [[[64, 46]]],
                }

            batches = list(
                iter_archive_dataset_batches(
                    plan,
                    shard_indices=[1],
                    batch_size=4,
                    seq_length=8,
                    batch_iterator=fake_batch_iterator,
                )
            )

        self.assertEqual(len(batches), 1)
        self.assertEqual(routed_gameids, [[2]])


if __name__ == "__main__":
    unittest.main()
