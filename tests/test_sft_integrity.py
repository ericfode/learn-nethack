from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from learn_nethack.action_manifest import ActionEntry, ActionManifest
from learn_nethack.sft_integrity import audit_sft_dataset, write_sft_integrity_report


def _row(*, split: str, task: str, gameid: int) -> dict:
    common = {
        "schema_version": "learn-nethack.sft-row.v1",
        "dataset_name": "fixture",
        "split": split,
        "task": task,
        "mode": "single_frame",
        "gameid": gameid,
        "episode_id": f"fixture:{gameid}",
        "step": 0,
        "sequence_id": f"fixture:{gameid}:100",
        "sequence_step": 0,
    }
    metadata = {
        "raw_key_code": 107,
        "raw_key_label": "k",
        "valid_action_ids": [0],
        "role": "Arc",
    }
    current = "MAP:\n@.\nMESSAGE:\nReady\nBLSTATS:\n[0]\nINVENTORY:\n<empty>"
    if task == "policy_action":
        return {
            **common,
            "messages": [
                {"role": "system", "content": "Policy"},
                {"role": "user", "content": f"Current observation:\n{current}"},
                {"role": "assistant", "content": '{"action_id": 0}'},
            ],
            "metadata": {**metadata, "target_action_id": 0},
        }
    following = "MAP:\n.@\nMESSAGE:\nMoved\nBLSTATS:\n[1]\nINVENTORY:\n<empty>"
    return {
        **common,
        "messages": [
            {"role": "system", "content": "Dynamics"},
            {
                "role": "user",
                "content": (
                    f'Action taken: {{"action_id": 0}}\nCurrent observation:\n{current}'
                ),
            },
            {"role": "assistant", "content": following},
        ],
        "metadata": {**metadata, "conditioning_action_id": 0},
    }


def _write_fixture(root: Path) -> None:
    ActionManifest(
        env_id="NetHackChallenge-v0",
        entries=(ActionEntry(0, "CompassDirection.N", 107, "k"),),
    ).save(root / "action_manifest.json")
    split_games = {"train": [1], "validation": [2], "test": [3]}
    (root / "split_manifest.json").write_text(
        json.dumps(split_games),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "learn-nethack.sft-manifest.v1",
                "env_id": "NetHackChallenge-v0",
                "tasks": ["policy_action", "next_frame"],
                "split_row_limits": {"train": 1, "validation": 1, "test": 1},
                "split_limits_satisfied": True,
                "accepted_rows_by_split": {
                    split: {"policy_action": 1, "next_frame": 1}
                    for split in split_games
                },
            }
        ),
        encoding="utf-8",
    )
    for split, gameids in split_games.items():
        rows = [
            _row(split=split, task=task, gameid=gameids[0])
            for task in ("policy_action", "next_frame")
        ]
        for task, row in zip(("policy_action", "next_frame"), rows, strict=True):
            (root / f"{split}.{task}.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


def test_sft_integrity_audit_proves_split_and_label_contracts() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)

        report = audit_sft_dataset(root, expected_env_id="NetHackChallenge-v0")
        path = write_sft_integrity_report(root / "integrity_report.json", report)
        written = json.loads(path.read_text(encoding="utf-8"))

    assert written["schema_version"] == "learn-nethack.sft-integrity.v2"
    assert written["passed"]
    assert written["failure_reasons"] == {}
    assert written["counts"]["policy_rows"] == 3
    assert written["counts"]["next_frame_rows"] == 3
    assert written["counts"]["paired_transition_rows"] == 3
    assert written["counts"]["paired_action_mismatches"] == 0
    assert written["actual_game_counts_by_split"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert written["action_distribution"]["dominant_action_rate"] == 1.0
    assert written["dynamics_diagnostics"]["changed_target_rate"] == 1.0


def test_sft_integrity_audit_rejects_split_leak_and_bad_true_key_label() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        validation_path = root / "validation.policy_action.jsonl"
        row = json.loads(validation_path.read_text(encoding="utf-8"))
        row["gameid"] = 1
        row["episode_id"] = "fixture:1"
        row["metadata"]["target_action_id"] = 9
        validation_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        report = audit_sft_dataset(root, expected_env_id="NetHackChallenge-v0")

    assert not report["passed"]
    assert report["failure_reasons"]["row_game_missing_from_split_manifest"] == 1
    assert report["failure_reasons"]["actual_game_overlap"] == 1
    assert report["failure_reasons"]["actual_episode_overlap"] == 1
    assert report["failure_reasons"]["policy_target_out_of_space"] == 1
    assert report["failure_reasons"]["policy_raw_key_target_mismatch"] == 1
    assert report["failure_reasons"]["policy_assistant_target_mismatch"] == 1


def test_sft_integrity_audit_rejects_environment_mismatch() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)

        report = audit_sft_dataset(root, expected_env_id="NetHack-v0")

    assert not report["passed"]
    assert report["failure_reasons"] == {"expected_env_mismatch": 1}


def test_sft_integrity_audit_requires_train_roles_in_each_heldout_split() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        train_path = root / "train.policy_action.jsonl"
        row = json.loads(train_path.read_text(encoding="utf-8"))
        row["metadata"]["role"] = "Wiz"
        train_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        report = audit_sft_dataset(root, expected_env_id="NetHackChallenge-v0")

    assert not report["passed"]
    assert report["failure_reasons"]["heldout_role_coverage_missing"] == 2
    assert report["role_coverage"]["missing_roles_by_split"] == {
        "validation": ["Wiz"],
        "test": ["Wiz"],
    }
