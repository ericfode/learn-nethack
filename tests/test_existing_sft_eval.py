from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from learn_nethack.action_manifest import ActionEntry, ActionManifest
from learn_nethack import existing_sft_eval
from learn_nethack.existing_sft_eval import load_existing_sft_eval_dataset


def _manifest() -> ActionManifest:
    return ActionManifest(
        env_id="NetHackChallenge-v0",
        entries=(
            ActionEntry(
                action_id=0,
                nle_action_name="CompassDirection.N",
                raw_key_code=107,
                key_label="k",
            ),
        ),
    )


def _write_dataset(root: Path) -> None:
    _manifest().save(root / "action_manifest.json")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "fixture",
                "env_id": "NetHackChallenge-v0",
                "mode": "single_frame",
            }
        ),
        encoding="utf-8",
    )
    for task in ("policy_action", "next_frame"):
        row = {
            "task": task,
            "split": "validation",
            "gameid": 42,
            "messages": [],
            "metadata": {},
        }
        (root / f"validation.{task}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )


def test_load_existing_sft_eval_dataset_reaudits_and_preserves_named_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_calls: list[tuple[Path, str]] = []

    def passing_audit(path, *, expected_env_id):
        audit_calls.append((Path(path), expected_env_id))
        return {"schema_version": "fixture", "status": "passed"}

    monkeypatch.setattr(existing_sft_eval, "audit_sft_dataset", passing_audit)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_dataset(root)

        dataset = load_existing_sft_eval_dataset(
            dataset_dir=root,
            split="validation",
            tasks=("policy_action", "next_frame"),
            mode="single_frame",
            action_manifest=_manifest(),
        )

    assert audit_calls == [(root, "NetHackChallenge-v0")]
    assert dataset.gameids == (42,)
    assert len(dataset.policy_rows) == 1
    assert len(dataset.next_frame_rows) == 1
    assert dataset.integrity_report["status"] == "passed"


def test_load_existing_sft_eval_dataset_fails_closed_on_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        existing_sft_eval,
        "audit_sft_dataset",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "failure_reasons": ["split_overlap"],
        },
    )
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_dataset(root)

        with pytest.raises(RuntimeError, match="split_overlap"):
            load_existing_sft_eval_dataset(
                dataset_dir=root,
                split="validation",
                tasks=("policy_action",),
                mode="single_frame",
                action_manifest=_manifest(),
            )


def test_load_existing_sft_eval_dataset_rejects_mode_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        existing_sft_eval,
        "audit_sft_dataset",
        lambda *_args, **_kwargs: {"status": "passed"},
    )
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_dataset(root)

        with pytest.raises(ValueError, match="mode mismatch"):
            load_existing_sft_eval_dataset(
                dataset_dir=root,
                split="validation",
                tasks=("policy_action",),
                mode="growing_context",
                action_manifest=_manifest(),
            )
