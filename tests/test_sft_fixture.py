from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from learn_nethack.sft_fixture import build_sft_overfit_fixture


def _row(task: str, index: int) -> dict:
    return {
        "task": task,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"input {index}"},
            {"role": "assistant", "content": f"target {index}"},
        ],
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_sft_overfit_fixture_writes_exact_balanced_rows() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source"
        source.mkdir()
        _write_rows(
            source / "train.policy_action.jsonl",
            [_row("policy_action", index) for index in range(6)],
        )
        _write_rows(
            source / "train.next_frame.jsonl",
            [_row("next_frame", index) for index in range(6)],
        )
        (source / "manifest.json").write_text(
            json.dumps({"dataset_name": "fixture", "label_source": "true_keypress"}),
            encoding="utf-8",
        )
        (source / "action_manifest.json").write_text("{}", encoding="utf-8")

        report = build_sft_overfit_fixture(
            source_dir=source,
            output_dir=root / "out",
            rows_per_task=4,
        )
        target = Path(report["output_dir"])
        combined = [
            json.loads(line)
            for line in (target / "train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        action_manifest_exists = (target / "action_manifest.json").exists()

    assert len(combined) == 8
    assert [row["task"] for row in combined] == [
        "policy_action",
        "next_frame",
    ] * 4
    assert manifest["accepted_policy_rows"] == 4
    assert manifest["accepted_next_frame_rows"] == 4
    assert manifest["label_source"] == "true_keypress"
    assert action_manifest_exists
