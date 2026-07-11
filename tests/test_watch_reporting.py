from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types

import pytest

from learn_nethack import modal_train, watch_reporting


class _FakeArtifact:
    def __init__(self, *, name: str, type: str):
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str, name: str | None = None) -> None:
        self.files.append(name or path)


class _FakeRun:
    id = "watch123"
    name = "watch-test"
    url = "https://wandb.example/watch123"
    path = ("entity", "project", "watch123")

    def __init__(self, calls: list[str]):
        self.calls = calls
        self.finished_exit_code: int | None = None

    def log(self, payload) -> None:
        self.calls.append(f"log:{sorted(payload)}")

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        self.calls.append(f"artifact:{artifact.name}")

    def finish(self, exit_code: int | None = None) -> None:
        self.finished_exit_code = exit_code
        self.calls.append(f"finish:{exit_code}")


def _watch_contract(root: Path) -> dict:
    watch_dir = root / "watch"
    return {
        "schema_version": "learn-nethack.watch-compare-contract.v1",
        "run_id": "watch-test",
        "watch": {"current_checkpoint": None},
        "artifacts": {
            "root": str(root),
            "watch_dir": str(watch_dir),
            "report": str(watch_dir / "report.json"),
            "contract": str(root / "reports" / "contract.json"),
        },
    }


def test_watch_initializes_wandb_before_rollout_and_records_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    fake_run = _FakeRun(calls)
    fake_wandb = types.ModuleType("wandb")

    def fake_init(**_kwargs):
        calls.append("wandb_init")
        return fake_run

    fake_wandb.init = fake_init
    fake_wandb.Artifact = _FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(modal_train, "_commit_mounted_volume", lambda _path: False)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        contract = _watch_contract(root)
        monkeypatch.setattr(
            modal_train,
            "local_watch_compare_contract",
            lambda **_kwargs: contract,
        )

        def fail_rollout(**_kwargs):
            calls.append("rollout")
            raise RuntimeError("fixture rollout failed")

        monkeypatch.setattr(modal_train, "run_checkpoint_compare", fail_rollout)

        with pytest.raises(RuntimeError, match="fixture rollout failed"):
            modal_train._watch_compare_impl(
                run_id="watch-test",
                action_manifest="fixture-manifest.json",
            )

        report = json.loads(
            Path(contract["artifacts"]["report"]).read_text(encoding="utf-8")
        )

    assert calls.index("wandb_init") < calls.index("rollout")
    assert report["status"] == "failed"
    assert report["failure_stage"] == "rollout"
    assert report["error_type"] == "RuntimeError"
    assert report["wandb"]["run_id"] == "watch123"
    assert fake_run.finished_exit_code == 1
    assert any(call == "artifact:watch-failure-watch-test" for call in calls)


def test_watch_sweep_replay_media_names_each_existing_seed_viewer() -> None:
    class FakeHtml:
        def __init__(self, path: str, *, inject: bool):
            self.path = path
            self.inject = inject

    fake_wandb = types.SimpleNamespace(Html=FakeHtml)
    with TemporaryDirectory() as tmp:
        first = Path(tmp) / "seed-1" / "index.html"
        second = Path(tmp) / "seed-2" / "index.html"
        first.parent.mkdir(parents=True)
        second.parent.mkdir(parents=True)
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")

        media = watch_reporting.watch_sweep_wandb_replay_media(
            fake_wandb,
            [
                {"seed": 1, "viewer_path": str(first)},
                {"seed": 2, "viewer_path": str(second)},
                {"seed": 3, "viewer_path": str(Path(tmp) / "missing.html")},
            ],
        )

    assert sorted(media) == [
        "watch_sweep/replay_seed_1",
        "watch_sweep/replay_seed_2",
    ]
    assert media["watch_sweep/replay_seed_1"].inject is False
