"""Metadata readers for local NLD ttyrec databases."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import sqlite3
from typing import Iterable


@dataclass(frozen=True)
class NldDbReport:
    db_path: str
    dataset_name: str
    root: str
    ttyrec_version: int
    game_count: int
    ttyrec_count: int


@dataclass(frozen=True)
class SftLabelReadinessReport:
    schema_version: str
    db_path: str
    dataset_name: str
    ttyrec_version: int
    game_count: int
    ttyrec_count: int
    status: str
    reason: str
    policy_action_trainable: bool
    next_frame_trainable: bool
    decoded_batch_keys: tuple[str, ...]
    required_decoded_keys: tuple[str, ...]


@dataclass(frozen=True)
class GameSplit:
    train: list[int]
    validation: list[int]
    test: list[int]


def inspect_nld_db(db_path: str | Path) -> NldDbReport:
    path = Path(db_path)
    with sqlite3.connect(path) as conn:
        root_row = conn.execute(
            "select dataset_name, root, ttyrec_version "
            "from roots order by dataset_name limit 1"
        ).fetchone()
        if root_row is None:
            raise ValueError(f"{path} has no roots row")
        game_count = int(conn.execute("select count(*) from games").fetchone()[0])
        ttyrec_count = int(conn.execute("select count(*) from ttyrecs").fetchone()[0])

    return NldDbReport(
        db_path=str(path),
        dataset_name=str(root_row[0]),
        root=str(root_row[1]),
        ttyrec_version=int(root_row[2]),
        game_count=game_count,
        ttyrec_count=ttyrec_count,
    )


def classify_sft_label_readiness(
    report: NldDbReport,
    *,
    decoded_batch_keys: Iterable[str] | None = None,
) -> SftLabelReadinessReport:
    """Classify whether an inspected NLD DB can produce supervised action rows."""
    keys = tuple(sorted(str(key) for key in (decoded_batch_keys or ())))
    key_set = set(keys)
    required_keys = ("keypresses", "actions")
    has_action_labels = bool(key_set & set(required_keys))
    if decoded_batch_keys is not None and not has_action_labels:
        status = "frame_only"
        reason = "decoded_action_labels_missing"
    elif has_action_labels:
        status = "labelled"
        reason = "decoded_action_labels_available"
    elif report.ttyrec_version >= 3:
        status = "probably_labelled"
        reason = "ttyrec_version_supports_action_labels_but_decode_not_sampled"
    else:
        status = "frame_only"
        reason = "ttyrec_version_lacks_action_labels"
    trainable = status == "labelled"
    return SftLabelReadinessReport(
        schema_version="learn-nethack.sft-label-readiness.v1",
        db_path=report.db_path,
        dataset_name=report.dataset_name,
        ttyrec_version=report.ttyrec_version,
        game_count=report.game_count,
        ttyrec_count=report.ttyrec_count,
        status=status,
        reason=reason,
        policy_action_trainable=trainable,
        next_frame_trainable=trainable,
        decoded_batch_keys=keys,
        required_decoded_keys=required_keys,
    )


def read_gameids(db_path: str | Path) -> list[int]:
    with sqlite3.connect(Path(db_path)) as conn:
        rows = conn.execute("select gameid from games order by gameid").fetchall()
    return [int(row[0]) for row in rows]


def read_game_metadata(db_path: str | Path) -> dict[int, dict]:
    query = (
        "select gameid, role, race, align, death, points, turns "
        "from games order by gameid"
    )
    with sqlite3.connect(Path(db_path)) as conn:
        rows = conn.execute(query).fetchall()
    return {
        int(gameid): {
            "role": role,
            "race": race,
            "align": align,
            "death": death,
            "points": points,
            "turns": turns,
        }
        for gameid, role, race, align, death, points, turns in rows
    }


def read_ttyrec_players(db_path: str | Path) -> list[str]:
    """Return sorted players that have at least one indexed ttyrec row."""
    path = Path(db_path)
    with sqlite3.connect(path) as conn:
        if _table_exists(conn, "ttyrec_players"):
            rows = conn.execute(
                "select distinct player from ttyrec_players order by player"
            ).fetchall()
        else:
            rows = conn.execute(
                "select distinct "
                "case when instr(path, '/') > 0 "
                "then substr(path, 1, instr(path, '/') - 1) "
                "else path end as player "
                "from ttyrecs order by player"
            ).fetchall()
    return [str(row[0]) for row in rows]


def copy_db_with_rewritten_root(
    *,
    source_db: str | Path,
    target_db: str | Path,
    new_root: str,
) -> dict[str, str]:
    """Copy an NLD DB and rewrite the copied roots table for a new filesystem."""
    source = Path(source_db)
    target = Path(target_db)
    if not source.exists():
        raise FileNotFoundError(f"NLD source DB does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    with sqlite3.connect(target) as conn:
        row_count = int(conn.execute("select count(*) from roots").fetchone()[0])
        if row_count == 0:
            raise ValueError(f"{target} has no roots row")
        conn.execute("update roots set root = ?", (new_root,))
        conn.commit()
    report = inspect_nld_db(target)
    return {
        "schema_version": "learn-nethack.nld-db-copy.v1",
        "source_db": str(source),
        "target_db": str(target),
        "target_root": report.root,
        "dataset_name": report.dataset_name,
    }


def ensure_ttyrec_player_index(db_path: str | Path) -> dict[str, int | str]:
    """Create an indexed player lookup table for fast shard DB filtering."""
    path = Path(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute("drop table if exists ttyrec_players")
        conn.execute(
            "create table ttyrec_players as "
            "select "
            "case when instr(path, '/') > 0 "
            "then substr(path, 1, instr(path, '/') - 1) "
            "else path end as player, "
            "path, part, size, mtime, gameid "
            "from ttyrecs"
        )
        conn.execute("create index idx_ttyrec_players_player on ttyrec_players(player)")
        conn.execute("create index idx_ttyrec_players_gameid on ttyrec_players(gameid)")
        indexed = int(conn.execute("select count(*) from ttyrec_players").fetchone()[0])
        players = int(
            conn.execute(
                "select count(distinct player) from ttyrec_players"
            ).fetchone()[0]
        )
        conn.commit()
    return {
        "schema_version": "learn-nethack.ttyrec-player-index.v1",
        "db_path": str(path),
        "indexed_ttyrecs": indexed,
        "indexed_players": players,
    }


def create_player_subset_db(
    *,
    source_db: str | Path,
    target_db: str | Path,
    player_names: Iterable[str],
    new_root: str,
) -> dict[str, int | str]:
    """Create a small NLD DB containing only selected players' ttyrecs."""
    source = Path(source_db)
    target = Path(target_db)
    players = sorted(set(player_names))
    if not players:
        raise ValueError("player_names must not be empty")
    if not source.exists():
        raise FileNotFoundError(f"NLD source DB does not exist: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        _copy_standard_nld_schema(src, dst)
        _copy_meta_and_roots(src, dst, new_root=new_root)
        dst.execute("create temp table selected_players (player text primary key)")
        dst.executemany(
            "insert into selected_players values (?)",
            [(player,) for player in players],
        )
        src_has_player_index = _table_exists(src, "ttyrec_players")
        ttyrec_rows = _selected_ttyrec_rows(
            src, players, use_index=src_has_player_index
        )
        dst.executemany(
            "insert into ttyrecs(path, part, size, mtime, gameid) "
            "values (?, ?, ?, ?, ?)",
            ttyrec_rows,
        )
        gameids = [
            int(row[0])
            for row in dst.execute(
                "select distinct gameid from ttyrecs order by gameid"
            ).fetchall()
        ]
        if gameids:
            placeholders = ",".join("?" for _ in gameids)
            game_rows = src.execute(
                f"select * from games where gameid in ({placeholders}) order by gameid",
                gameids,
            ).fetchall()
            dataset_rows = src.execute(
                f"select * from datasets where gameid in ({placeholders}) "
                "order by gameid",
                gameids,
            ).fetchall()
            dst.executemany(
                f"insert into games values ({','.join('?' for _ in game_rows[0])})",
                game_rows,
            )
            if dataset_rows:
                dst.executemany("insert into datasets values (?, ?)", dataset_rows)
        dst.commit()
        selected_ttyrec_count = int(
            dst.execute("select count(*) from ttyrecs").fetchone()[0]
        )
        selected_game_count = int(
            dst.execute("select count(*) from games").fetchone()[0]
        )

    return {
        "schema_version": "learn-nethack.nld-player-subset-db.v1",
        "source_db": str(source),
        "target_db": str(target),
        "target_root": new_root,
        "selected_player_count": len(players),
        "selected_game_count": selected_game_count,
        "selected_ttyrec_count": selected_ttyrec_count,
    }


def _copy_standard_nld_schema(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    for table_name in ("meta", "ttyrecs", "games", "datasets", "roots"):
        row = src.execute(
            "select sql from sqlite_master where type = 'table' and name = ?",
            (table_name,),
        ).fetchone()
        if row is None:
            raise ValueError(f"source DB has no {table_name} table")
        dst.execute(str(row[0]))


def _copy_meta_and_roots(
    src: sqlite3.Connection, dst: sqlite3.Connection, *, new_root: str
) -> None:
    dst.executemany("insert into meta values (?, ?)", src.execute("select * from meta"))
    root_rows = [
        (dataset_name, new_root, ttyrec_version)
        for dataset_name, _old_root, ttyrec_version in src.execute(
            "select * from roots"
        )
    ]
    dst.executemany("insert into roots values (?, ?, ?)", root_rows)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _selected_ttyrec_rows(
    conn: sqlite3.Connection, players: list[str], *, use_index: bool
) -> list[tuple]:
    table = "ttyrec_players" if use_index else "ttyrecs"
    player_expr = (
        "player"
        if use_index
        else "case when instr(path, '/') > 0 "
        "then substr(path, 1, instr(path, '/') - 1) else path end"
    )
    conn.execute(
        "create temp table if not exists selected_players (player text primary key)"
    )
    conn.execute("delete from selected_players")
    conn.executemany(
        "insert into selected_players values (?)", [(player,) for player in players]
    )
    return conn.execute(
        f"select path, part, size, mtime, gameid from {table} "
        f"where {player_expr} in (select player from selected_players) "
        "order by gameid, part, path"
    ).fetchall()


def _bucket(gameid: int, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{gameid}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_gameids(
    gameids: list[int],
    *,
    seed: int,
    train_ratio: float = 0.90,
    validation_ratio: float = 0.05,
) -> GameSplit:
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for gameid in sorted(gameids):
        value = _bucket(gameid, seed)
        if value < train_ratio:
            train.append(gameid)
        elif value < train_ratio + validation_ratio:
            validation.append(gameid)
        else:
            test.append(gameid)
    return GameSplit(train=train, validation=validation, test=test)


def order_gameids_for_split_role_coverage(
    gameids: list[int],
    *,
    game_metadata_by_id: dict[int, dict],
    seed: int,
) -> list[int]:
    """Round-robin split and role so capped builds do not fill from few games."""
    splits = split_gameids(gameids, seed=seed)
    split_by_gameid = {
        int(gameid): split
        for split, values in (
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        )
        for gameid in values
    }
    grouped: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for gameid in sorted(gameids, key=lambda value: (_bucket(value, seed + 1), value)):
        split = split_by_gameid[int(gameid)]
        role = str(game_metadata_by_id.get(int(gameid), {}).get("role") or "<missing>")
        grouped[(split, role)].append(int(gameid))

    roles = sorted({role for _split, role in grouped})
    split_order = ("validation", "test", "train")
    ordered: list[int] = []
    while grouped:
        made_progress = False
        for role in roles:
            for split in split_order:
                key = (split, role)
                values = grouped.get(key)
                if not values:
                    continue
                ordered.append(values.popleft())
                made_progress = True
                if not values:
                    del grouped[key]
        if not made_progress:
            raise RuntimeError("failed to order game IDs for split and role coverage")

    if sorted(ordered) != sorted(gameids):
        raise RuntimeError("split-role game ordering lost or duplicated game IDs")
    return ordered
