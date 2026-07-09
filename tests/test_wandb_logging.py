from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from learn_nethack.cli import app
from learn_nethack.wandb_logging import log_sft_build_to_wandb, resolve_local_wandb_mode
from learn_nethack.wandb_logging import build_wandb_visibility_report


class WandbLoggingTests(unittest.TestCase):
    def test_explicit_offline_mode_is_allowed_without_api_key(self) -> None:
        self.assertEqual(resolve_local_wandb_mode({"WANDB_MODE": "offline"}), "offline")

    def test_online_mode_requires_api_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY is required"):
            resolve_local_wandb_mode({})

    def test_online_mode_with_api_key(self) -> None:
        self.assertEqual(
            resolve_local_wandb_mode({"WANDB_API_KEY": "present"}), "online"
        )

    def test_sft_build_logging_records_metrics_and_artifact(self) -> None:
        fake_wandb = types.ModuleType("wandb")
        recorded: dict[str, object] = {}

        class FakeArtifact:
            def __init__(self, name: str, type: str):
                self.name = name
                self.type = type
                self.files: list[str] = []

            def add_file(self, path: str) -> None:
                self.files.append(Path(path).name)

        class FakeRun:
            def __init__(self) -> None:
                self.logged: list[dict[str, object]] = []
                self.artifacts: list[FakeArtifact] = []
                self.finished = False

            def log(self, payload: dict[str, object]) -> None:
                self.logged.append(payload)

            def log_artifact(self, artifact: FakeArtifact) -> None:
                self.artifacts.append(artifact)

            def finish(self) -> None:
                self.finished = True

        fake_run = FakeRun()

        def fake_init(**kwargs):
            recorded["init_kwargs"] = kwargs
            return fake_run

        class FakeSettings:
            def __init__(self, **kwargs) -> None:
                recorded["settings_kwargs"] = kwargs

        fake_wandb.init = fake_init
        fake_wandb.Artifact = FakeArtifact
        fake_wandb.Settings = FakeSettings
        original_wandb = sys.modules.get("wandb")
        sys.modules["wandb"] = fake_wandb
        try:
            with TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                (out_dir / "manifest.json").write_text("{}", encoding="utf-8")
                (out_dir / "rejection_report.json").write_text("{}", encoding="utf-8")
                mode = log_sft_build_to_wandb(
                    output_dir=out_dir,
                    metrics={
                        "accepted_policy_rows": 64,
                        "accepted_next_frame_rows": 63,
                        "rejected_rows": 4,
                    },
                    config={"dataset_name": "nld-aa-taster", "mode": "single_frame"},
                    env={"WANDB_MODE": "offline"},
                    project="learn-nethack-test",
                    run_name="sft-build-smoke",
                )
        finally:
            if original_wandb is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = original_wandb

        self.assertEqual(mode, "offline")
        self.assertEqual(recorded["init_kwargs"]["mode"], "offline")
        self.assertEqual(recorded["init_kwargs"]["job_type"], "sft-data-build")
        self.assertEqual(
            recorded["settings_kwargs"],
            {"x_disable_machine_info": True, "x_disable_stats": True},
        )
        self.assertEqual(
            fake_run.logged,
            [
                {
                    "sft_data/accepted_next_frame_rows": 63,
                    "sft_data/accepted_policy_rows": 64,
                    "sft_data/rejected_rows": 4,
                }
            ],
        )
        self.assertTrue(fake_run.finished)
        self.assertEqual(fake_run.artifacts[0].type, "dataset")
        self.assertEqual(
            fake_run.artifacts[0].files,
            ["manifest.json", "rejection_report.json"],
        )

    def test_wandb_visibility_report_finds_offline_runs_and_sync_command(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            root_run = root / "wandb" / "offline-run-20260615_160831-f9p62op3"
            artifact_run = (
                root
                / "artifacts"
                / "sft"
                / "example"
                / "wandb"
                / "offline-run-20260615_161125-5d5vxext"
            )
            ignored = root / "wandb" / "offline-run-20260615_empty"
            for run_dir, run_file in (
                (root_run, "run-f9p62op3.wandb"),
                (artifact_run, "run-5d5vxext.wandb"),
            ):
                run_dir.mkdir(parents=True)
                (run_dir / run_file).write_text("fixture", encoding="utf-8")
            ignored.mkdir(parents=True)

            report = build_wandb_visibility_report(root=root, env={})

        self.assertEqual(report["offline_run_count"], 2)
        self.assertEqual(
            report["offline_run_paths"],
            [str(artifact_run), str(root_run)],
        )
        self.assertEqual(report["api_key_configured"], False)
        self.assertEqual(
            report["sync_command"],
            f"uv run wandb sync {artifact_run} {root_run}",
        )
        self.assertIn("WANDB_API_KEY", report["recommended_next_step"])

    def test_cli_wandb_status_reports_offline_visibility(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "wandb" / "offline-run-20260615_160831-f9p62op3"
            run_dir.mkdir(parents=True)
            (run_dir / "run-f9p62op3.wandb").write_text("fixture", encoding="utf-8")

            result = runner.invoke(app, ["wandb", "status", "--root", str(root)])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = types.SimpleNamespace(**json.loads(result.output))
        self.assertEqual(payload.offline_run_count, 1)
        self.assertEqual(payload.sync_command, f"uv run wandb sync {run_dir}")


if __name__ == "__main__":
    unittest.main()
