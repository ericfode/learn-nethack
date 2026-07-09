from __future__ import annotations

import sys
import sqlite3
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from learn_nethack.nld_index import build_altorg_index, prepare_altorg_staging_root
from learn_nethack.nld_metadata import (
    copy_db_with_rewritten_root,
    create_player_subset_db,
    ensure_ttyrec_player_index,
    inspect_nld_db,
)


class NldIndexTests(unittest.TestCase):
    def test_prepare_altorg_staging_root_links_metadata_and_player_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_root = root / "metadata"
            ttyrec_root = root / "ttyrecs"
            staging_root = root / "staging"
            metadata_root.mkdir()
            ttyrec_root.mkdir()
            (metadata_root / "xlogfile.full.txt").write_text("xlog", encoding="utf-8")
            (metadata_root / "blacklist.txt").write_text("", encoding="utf-8")
            (ttyrec_root / "PlayerOne").mkdir()
            (ttyrec_root / "PlayerOne" / "2020-01-01.00:00:00.ttyrec.bz2").write_text(
                "ttyrec",
                encoding="utf-8",
            )

            report = prepare_altorg_staging_root(
                metadata_root=metadata_root,
                ttyrec_root=ttyrec_root,
                staging_root=staging_root,
            )

            self.assertEqual(report["xlogfile_count"], 1)
            self.assertEqual(report["player_dir_count"], 1)
            self.assertTrue((staging_root / "xlogfile.full.txt").is_symlink())
            self.assertTrue((staging_root / "blacklist.txt").is_symlink())
            self.assertTrue((staging_root / "PlayerOne").is_symlink())

    def test_build_altorg_index_uses_nle_populate_db(self) -> None:
        fake_nle = types.ModuleType("nle")
        fake_dataset = types.ModuleType("nle.dataset")
        fake_db = types.SimpleNamespace()
        fake_populate_db = types.SimpleNamespace()
        calls: list[tuple] = []

        def fake_create(*, filename: str) -> None:
            calls.append(("create", filename))
            Path(filename).write_text("db", encoding="utf-8")

        def fake_add_altorg_directory(path: str, name: str, filename: str) -> None:
            calls.append(("add_altorg_directory", path, name, filename))

        fake_db.create = fake_create
        fake_populate_db.add_altorg_directory = fake_add_altorg_directory
        fake_dataset.db = fake_db
        fake_dataset.populate_db = fake_populate_db
        fake_nle.dataset = fake_dataset
        original_nle = sys.modules.get("nle")
        original_dataset = sys.modules.get("nle.dataset")
        sys.modules["nle"] = fake_nle
        sys.modules["nle.dataset"] = fake_dataset

        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                metadata_root = root / "metadata"
                ttyrec_root = root / "ttyrecs"
                staging_root = root / "staging"
                db_path = root / "indexed" / "ttyrecs.db"
                metadata_root.mkdir()
                ttyrec_root.mkdir()
                (metadata_root / "xlogfile.full.txt").write_text(
                    "xlog",
                    encoding="utf-8",
                )
                (metadata_root / "blacklist.txt").write_text("", encoding="utf-8")
                (ttyrec_root / "PlayerOne").mkdir()

                report = build_altorg_index(
                    metadata_root=metadata_root,
                    ttyrec_root=ttyrec_root,
                    staging_root=staging_root,
                    db_path=db_path,
                    dataset_name="nld-nao",
                )
        finally:
            if original_nle is None:
                sys.modules.pop("nle", None)
            else:
                sys.modules["nle"] = original_nle
            if original_dataset is None:
                sys.modules.pop("nle.dataset", None)
            else:
                sys.modules["nle.dataset"] = original_dataset

        self.assertEqual(report["dataset_name"], "nld-nao")
        self.assertEqual(report["db_path"], str(db_path))
        self.assertEqual(calls[0], ("create", str(db_path)))
        self.assertEqual(
            calls[1],
            ("add_altorg_directory", str(staging_root), "nld-nao", str(db_path)),
        )

    def test_copy_db_with_rewritten_root_preserves_source_and_updates_copy(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ttyrecs.db"
            target = root / "modal" / "ttyrecs.db"
            with sqlite3.connect(source) as conn:
                conn.execute(
                    "create table roots "
                    "(dataset_name text, root text, ttyrec_version integer)"
                )
                conn.execute(
                    "create table games "
                    "(gameid integer, role text, race text, align text, death text, "
                    "points integer, turns integer)"
                )
                conn.execute("create table ttyrecs (gameid integer)")
                conn.execute(
                    "insert into roots values ('nld-aa-taster', '/local/root', 3)"
                )
                conn.execute(
                    "insert into games values (1, 'Val', 'Hum', 'Law', 'escaped', 1, 2)"
                )
                conn.execute("insert into ttyrecs values (1)")

            report = copy_db_with_rewritten_root(
                source_db=source,
                target_db=target,
                new_root="/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
            )

            source_report = inspect_nld_db(source)
            target_report = inspect_nld_db(target)

        self.assertEqual(source_report.root, "/local/root")
        self.assertEqual(
            target_report.root,
            "/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
        )
        self.assertEqual(report["schema_version"], "learn-nethack.nld-db-copy.v1")
        self.assertEqual(report["target_root"], target_report.root)

    def test_ensure_ttyrec_player_index_adds_player_lookup_table(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ttyrecs.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "create table ttyrecs "
                    "(path text, part integer, size integer, mtime real, gameid integer)"
                )
                conn.execute(
                    "insert into ttyrecs values ('Alice/one.ttyrec.bz2', 0, 10, 0.0, 1)"
                )
                conn.execute(
                    "insert into ttyrecs values ('Bob/two.ttyrec.bz2', 0, 20, 0.0, 2)"
                )

            report = ensure_ttyrec_player_index(db_path)

            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "select player, path, gameid from ttyrec_players order by player"
                ).fetchall()

        self.assertEqual(report["indexed_ttyrecs"], 2)
        self.assertEqual(
            rows,
            [
                ("Alice", "Alice/one.ttyrec.bz2", 1),
                ("Bob", "Bob/two.ttyrec.bz2", 2),
            ],
        )

    def test_create_player_subset_db_filters_rows_and_rewrites_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ttyrecs.db"
            target = root / "subset" / "ttyrecs.db"
            with sqlite3.connect(source) as conn:
                conn.execute("create table meta (ctime real, mtime real)")
                conn.execute(
                    "create table roots "
                    "(dataset_name text primary key, root text, ttyrec_version integer)"
                )
                conn.execute(
                    "create table games "
                    "(gameid integer primary key, role text, race text, align text, "
                    "death text, points integer, turns integer)"
                )
                conn.execute(
                    "create table datasets "
                    "(gameid integer, dataset_name text, primary key(dataset_name, gameid))"
                )
                conn.execute(
                    "create table ttyrecs "
                    "(path text, part integer, size integer, mtime real, gameid integer, "
                    "primary key(gameid, part, path))"
                )
                conn.execute("insert into meta values (1.0, 2.0)")
                conn.execute("insert into roots values ('nld-nao', '/old/root', 1)")
                for gameid, player in ((1, "Alice"), (2, "Bob"), (3, "Alice")):
                    conn.execute(
                        "insert into games values (?, 'Val', 'Hum', 'Law', 'quit', 1, 2)",
                        (gameid,),
                    )
                    conn.execute(
                        "insert into datasets values (?, 'nld-nao')", (gameid,)
                    )
                    conn.execute(
                        "insert into ttyrecs values (?, 0, 10, 0.0, ?)",
                        (f"{player}/{gameid}.ttyrec.bz2", gameid),
                    )
            ensure_ttyrec_player_index(source)

            report = create_player_subset_db(
                source_db=source,
                target_db=target,
                player_names=["Alice"],
                new_root="/tmp/nld-shard",
            )

            subset_report = inspect_nld_db(target)
            with sqlite3.connect(target) as conn:
                gameids = [
                    row[0]
                    for row in conn.execute("select gameid from games order by gameid")
                ]
                ttyrec_paths = [
                    row[0]
                    for row in conn.execute("select path from ttyrecs order by path")
                ]
                datasets = conn.execute(
                    "select * from datasets order by gameid"
                ).fetchall()

        self.assertEqual(report["selected_player_count"], 1)
        self.assertEqual(report["selected_game_count"], 2)
        self.assertEqual(report["selected_ttyrec_count"], 2)
        self.assertEqual(subset_report.root, "/tmp/nld-shard")
        self.assertEqual(gameids, [1, 3])
        self.assertEqual(ttyrec_paths, ["Alice/1.ttyrec.bz2", "Alice/3.ttyrec.bz2"])
        self.assertEqual(datasets, [(1, "nld-nao"), (3, "nld-nao")])


if __name__ == "__main__":
    unittest.main()
