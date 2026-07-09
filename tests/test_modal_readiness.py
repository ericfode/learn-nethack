import importlib
import json
import os
import sqlite3
import tarfile
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from learn_nethack.cli import app
from learn_nethack.modal_upload import build_player_tar_shard


ROOT = Path(__file__).resolve().parents[1]


def _make_archive_fixture(root: Path) -> Path:
    source = root / "source"
    ttyrec = source / "ana" / "a.ttyrec.bz2"
    ttyrec.parent.mkdir(parents=True)
    ttyrec.write_text("ttyrec fixture", encoding="utf-8")
    shard = root / "shard.tar"
    with tarfile.open(shard, "w") as archive:
        archive.add(ttyrec.parent, arcname="ana", recursive=True)

    db = root / "ttyrecs.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "create table roots (dataset_name text primary key, root text, ttyrec_version integer)"
        )
        conn.execute(
            "create table games (gameid integer primary key, role text, race text, align text, death text, points integer, turns integer)"
        )
        conn.execute(
            "create table ttyrecs (path text, part integer, size integer, mtime real, gameid integer)"
        )
        conn.execute("insert into roots values ('fixture-nld', '/tmp/stale', 3)")
        conn.execute(
            "insert into games values (1, 'Sam', 'Hum', 'Law', 'quit', 10, 20)"
        )
        conn.execute("insert into ttyrecs values ('ana/a.ttyrec.bz2', 0, 100, 0.0, 1)")

    manifest = root / "archive.jsonl"
    manifest.write_text(
        json.dumps({"shard_tar": str(shard), "shard_db": str(db)}) + "\n",
        encoding="utf-8",
    )
    return manifest


class ModalReadinessTests(unittest.TestCase):
    def _import_modal_config(self):
        try:
            return importlib.import_module("learn_nethack.modal_config")
        except ModuleNotFoundError as exc:
            self.fail(f"learn_nethack.modal_config must exist: {exc}")

    def test_wandb_is_core_dependency_and_balrog_is_not_core(self) -> None:
        pyproject_path = ROOT / "pyproject.toml"

        self.assertTrue(pyproject_path.exists(), "pyproject.toml must exist")

        pyproject = tomllib.loads(pyproject_path.read_text())

        core_dependencies = pyproject["project"]["dependencies"]
        modal_dependencies = pyproject["project"]["optional-dependencies"][
            "modal-train"
        ]

        self.assertTrue(
            any(dep.startswith("wandb") for dep in core_dependencies),
            "wandb must be a core dependency, not an optional reporting extra",
        )
        self.assertFalse(
            any("balrog" in dep.lower() for dep in core_dependencies),
            "BALROG must not enter the core trainer dependency path",
        )
        self.assertFalse(
            any("balrog" in dep.lower() for dep in modal_dependencies),
            "BALROG must stay isolated from the Modal training image",
        )

    def test_modal_config_declares_resources_and_artifact_layout(self) -> None:
        config = self._import_modal_config()

        volume_names = {volume.name for volume in config.MODAL_VOLUMES}

        self.assertEqual(
            volume_names,
            {
                "learn-nethack-datasets",
                "learn-nethack-runs",
                "learn-nethack-hf-cache",
                "learn-nethack-watch",
            },
        )
        self.assertEqual(tuple(config.SECRET_ENV_VARS), ("HF_TOKEN", "WANDB_API_KEY"))
        self.assertEqual(tuple(config.MODAL_SECRET_NAMES), ("hf-token", "wandb-secret"))
        self.assertEqual(tuple(config.MODAL_SOURCE_MODULES), ("learn_nethack",))
        self.assertEqual(config.HF_CACHE_MOUNT_PATH, "/cache/huggingface")
        self.assertEqual(
            config.modal_hf_cache_env(),
            {
                "HF_HOME": "/cache/huggingface",
                "HF_HUB_CACHE": "/cache/huggingface/hub",
                "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
                "TRANSFORMERS_CACHE": "/cache/huggingface",
            },
        )
        self.assertIn("nle>=1.3.0", config.MODAL_TRAIN_PIP_PACKAGES)
        self.assertTrue(
            any(
                "github.com/unslothai/unsloth.git" in package
                for package in config.MODAL_TRAIN_PIP_PACKAGES
            ),
            "Gemma 4 training requires current Unsloth from git.",
        )
        self.assertTrue(
            any(
                "github.com/unslothai/unsloth-zoo.git" in package
                for package in config.MODAL_TRAIN_PIP_PACKAGES
            ),
            "Gemma 4 training requires current Unsloth Zoo from git.",
        )

        layout = config.run_artifact_layout("modal-readiness-smoke")

        self.assertEqual(
            layout["report"],
            "/runs/modal-readiness-smoke/reports/modal_readiness_report.json",
        )
        self.assertEqual(layout["wandb"], "/runs/modal-readiness-smoke/wandb")
        self.assertEqual(layout["adapter"], "/runs/modal-readiness-smoke/adapters")
        self.assertEqual(layout["ttyrec"], "/runs/modal-readiness-smoke/ttyrec")
        self.assertEqual(layout["replay"], "/runs/modal-readiness-smoke/replay")

    def test_hf_token_alias_normalizer_preserves_existing_token(self) -> None:
        config = self._import_modal_config()
        env = {"HF_TOKEN": "primary", "HUGGING_FACE_HUB_TOKEN": "alias"}

        changed = config.normalize_hf_token_env(env)

        self.assertFalse(changed)
        self.assertEqual(env["HF_TOKEN"], "primary")

    def test_hf_token_alias_normalizer_sets_hf_token_from_common_alias(self) -> None:
        config = self._import_modal_config()
        env = {"HUGGING_FACE_HUB_TOKEN": "alias-token"}

        changed = config.normalize_hf_token_env(env)

        self.assertTrue(changed)
        self.assertEqual(env["HF_TOKEN"], "alias-token")

    def test_modal_hf_cache_commit_is_local_noop(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(modal_train._commit_mounted_volume("/cache/huggingface"))

    def test_wandb_gate_requires_explicit_offline_without_api_key(self) -> None:
        config = self._import_modal_config()

        with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY is required"):
            config.resolve_wandb_mode({})

        offline = config.resolve_wandb_mode({"WANDB_MODE": "offline"})
        online = config.resolve_wandb_mode({"WANDB_API_KEY": "present"})

        self.assertEqual(offline.mode, "offline")
        self.assertFalse(offline.requires_api_key)
        self.assertEqual(online.mode, "online")
        self.assertTrue(online.requires_api_key)

    def test_modal_smoke_commands_are_exact_and_do_not_embed_secrets(self) -> None:
        config = self._import_modal_config()

        commands = config.smoke_commands("modal-readiness-smoke")

        self.assertEqual(
            commands["offline_readiness"],
            "WANDB_MODE=offline modal run "
            "src/learn_nethack/modal_train.py::readiness "
            "--run-id modal-readiness-smoke",
        )
        self.assertEqual(
            commands["online_readiness"],
            "modal run src/learn_nethack/modal_train.py::readiness "
            "--run-id modal-readiness-smoke",
        )
        self.assertNotIn("WANDB_API_KEY=", " ".join(commands.values()))
        self.assertNotIn("HF_TOKEN=", " ".join(commands.values()))

    def test_offline_modal_smoke_does_not_require_training_secret(self) -> None:
        config = self._import_modal_config()

        offline_env = {"WANDB_MODE": "offline"}

        self.assertEqual(config.modal_secret_names_for_env(offline_env), ())
        self.assertEqual(
            config.modal_function_env(offline_env), {"WANDB_MODE": "offline"}
        )
        self.assertEqual(
            config.modal_secret_names_for_env({}),
            ("hf-token", "wandb-secret"),
        )
        self.assertEqual(config.modal_function_env({}), {})

    def test_full_run_commands_name_upload_train_eval_and_compare_steps(self) -> None:
        config = self._import_modal_config()

        commands = config.full_run_commands(
            run_id="full-sft",
            local_dataset_root="/Users/ericfode/data/nld/nld-aa-taster",
        )

        self.assertEqual(
            sorted(commands),
            [
                "compare",
                "eval_baseline",
                "eval_trained",
                "train",
                "upload_action_manifest",
                "upload_dataset",
            ],
        )
        self.assertIn(
            "modal volume put learn-nethack-datasets "
            "/Users/ericfode/data/nld/nld-aa-taster /nld/",
            commands["upload_dataset"],
        )
        self.assertEqual(
            commands["upload_action_manifest"],
            "modal volume put --force learn-nethack-datasets "
            "artifacts/action_manifest.json /action_manifest.json",
        )
        self.assertIn(
            "modal run src/learn_nethack/modal_train.py::sft_train",
            commands["train"],
        )
        self.assertIn(
            "--nle-root /datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
            commands["train"],
        )
        self.assertIn(
            "modal run src/learn_nethack/modal_train.py::sft_eval",
            commands["eval_trained"],
        )
        self.assertIn(
            "modal run src/learn_nethack/modal_train.py::sft_compare",
            commands["compare"],
        )

    def test_watch_compare_commands_name_upload_and_modal_run(self) -> None:
        config = self._import_modal_config()

        commands = config.watch_compare_commands(run_id="watch-10")

        self.assertEqual(
            sorted(commands),
            ["run_watch_compare", "upload_action_manifest"],
        )
        self.assertEqual(
            commands["upload_action_manifest"],
            "modal volume put learn-nethack-datasets "
            "artifacts/action_manifest.json /action_manifest.json",
        )
        self.assertEqual(
            commands["run_watch_compare"],
            "WANDB_MODE=offline modal run "
            "src/learn_nethack/modal_train.py::watch_compare "
            "--run-id watch-10 "
            "--action-manifest /datasets/action_manifest.json "
            "--env-id NetHack-v0 "
            "--model-name google/gemma-4-E2b-it "
            "--max-steps 10",
        )

    def test_archive_full_run_commands_use_archive_manifest_not_unpacked_db(
        self,
    ) -> None:
        config = self._import_modal_config()

        commands = config.archive_full_run_commands(
            run_id="full-archive-sft",
            archive_manifest_path="artifacts/nld-nao-archive.jsonl",
        )

        self.assertEqual(
            sorted(commands),
            [
                "compare",
                "eval_baseline",
                "eval_trained",
                "train",
                "upload_action_manifest",
                "upload_archive_manifest",
            ],
        )
        self.assertIn(
            "modal volume put learn-nethack-datasets "
            "artifacts/nld-nao-archive.jsonl /nld-nao-archive.jsonl",
            commands["upload_archive_manifest"],
        )
        self.assertIn(
            "--archive-manifest /datasets/nld-nao-archive.jsonl",
            commands["train"],
        )
        self.assertNotIn("--db", commands["train"])
        self.assertNotIn("--nle-root", commands["train"])
        self.assertIn(
            "--archive-manifest /datasets/nld-nao-archive.jsonl",
            commands["eval_baseline"],
        )

    def test_sft_existing_dataset_followup_commands_name_pull_and_train_steps(
        self,
    ) -> None:
        config = self._import_modal_config()

        commands = config.sft_existing_dataset_followup_commands(
            build_run_id="full-build",
            train_run_id="full-train",
            app_id="ap-123",
            max_steps=250,
            training_objective="policy_only",
            eval_mode="feedback_context_6",
        )

        self.assertEqual(commands["logs"], "modal app logs ap-123")
        self.assertEqual(
            commands["pull_build_report"],
            "modal volume get learn-nethack-runs "
            "full-build/reports/sft_build_report.json "
            "artifacts/full-build/sft_build_report.json",
        )
        self.assertIn(
            "src/learn_nethack/modal_train.py::sft_train_existing",
            commands["train_existing"],
        )
        self.assertIn(
            "--dataset-dir /runs/full-build/sft-data",
            commands["train_existing"],
        )
        self.assertIn("--run-id full-train", commands["train_existing"])
        self.assertIn("--max-steps 250", commands["train_existing"])
        self.assertIn("--training-objective policy_only", commands["train_existing"])
        self.assertIn(
            "--eval-tasks policy_action,next_frame",
            commands["eval_baseline_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-eval-mode both",
            commands["eval_baseline_policy_and_next_frame"],
        )
        self.assertIn(
            "--mode feedback_context_6",
            commands["eval_baseline_policy_and_next_frame"],
        )
        self.assertIn(
            "--mode feedback_context_6",
            commands["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-max-new-tokens 512",
            commands["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-generate-max-rows 64",
            commands["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-sequence-horizons 1,5,10",
            commands["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-sequence-max-windows 64",
            commands["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--adapter /runs/full-train/adapters",
            commands["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "/runs/full-train-baseline-eval/reports/sft_eval_metrics.json",
            commands["compare_policy_and_next_frame"],
        )
        self.assertIn(
            "/runs/full-train-trained-eval/reports/sft_eval_metrics.json",
            commands["compare_policy_and_next_frame"],
        )
        self.assertIn(
            "src/learn_nethack/modal_train.py::watch_compare_sweep",
            commands["watch_compare_score_damage"],
        )
        self.assertIn(
            "--action-manifest /datasets/action_manifest_nethack_v0.json",
            commands["watch_compare_score_damage"],
        )
        self.assertIn("--env-id NetHack-v0", commands["watch_compare_score_damage"])
        self.assertIn(
            "--character mon-hum-neu-mal",
            commands["watch_compare_score_damage"],
        )
        self.assertIn(
            "--seeds 20260615,20260616,20260617,20260618,20260619,20260620,20260621,20260622,20260623,20260624,20260625,20260626,20260627,20260628,20260629,20260630",
            commands["watch_compare_score_damage"],
        )
        self.assertIn(
            "--current-checkpoint /runs/full-train/adapters",
            commands["watch_compare_score_damage"],
        )
        self.assertIn("--max-steps 80", commands["watch_compare_score_damage"])
        self.assertIn(
            "full-train/reports/score_to_beat_policy_and_next_frame.json",
            commands["pull_score_to_beat_policy_and_next_frame"],
        )
        self.assertIn(
            "full-train-watch-score-damage/sweep_report.json",
            commands["pull_watch_report"],
        )
        self.assertIn(
            "uv run nethack-gemma sft proof-gate",
            commands["proof_gate_policy_next_frame_and_watch"],
        )
        self.assertIn(
            "artifacts/full-train/score_to_beat_policy_and_next_frame.json",
            commands["proof_gate_policy_next_frame_and_watch"],
        )
        self.assertIn(
            "artifacts/full-train/training_proof_gate.json",
            commands["proof_gate_policy_next_frame_and_watch"],
        )

    def test_full_build_followup_cli_forwards_next_n_eval_options(self) -> None:
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "sft",
                "full-build-followup",
                "--build-run-id",
                "full-build",
                "--train-run-id",
                "full-train",
                "--baseline-eval-run-id",
                "baseline-seq64",
                "--trained-eval-run-id",
                "trained-seq64",
                "--eval-mode",
                "feedback_context_6",
                "--next-frame-sequence-max-windows",
                "32",
                "--next-frame-generate-max-rows",
                "24",
                "--next-frame-max-new-tokens",
                "128",
                "--watch-run-id",
                "watch-score-damage",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertIn(
            "--run-id baseline-seq64",
            payload["commands"]["eval_baseline_policy_and_next_frame"],
        )
        self.assertIn(
            "--run-id trained-seq64",
            payload["commands"]["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-sequence-max-windows 32",
            payload["commands"]["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--mode feedback_context_6",
            payload["commands"]["eval_baseline_policy_and_next_frame"],
        )
        self.assertIn(
            "--mode feedback_context_6",
            payload["commands"]["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-generate-max-rows 24",
            payload["commands"]["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--next-frame-max-new-tokens 128",
            payload["commands"]["eval_trained_policy_and_next_frame"],
        )
        self.assertIn(
            "--run-id watch-score-damage",
            payload["commands"]["watch_compare_score_damage"],
        )
        self.assertIn(
            "artifacts/watch/watch-score-damage/sweep_report.json",
            payload["commands"]["proof_gate_policy_next_frame_and_watch"],
        )
        self.assertEqual(payload["training_gate"]["status"], "closed")
        self.assertFalse(payload["training_gate"]["train_ready"])
        self.assertEqual(
            payload["training_gate"]["missing_markers"],
            ["train_jsonl", "manifest", "rejection_report", "sft_build_report"],
        )
        self.assertEqual(
            payload["training_gate"]["next_action"],
            "wait for or rerun full-dataset build before training",
        )

    def test_full_build_followup_uses_existing_remote_checked_status(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_dir = root / "full-build"
            build_dir.mkdir(parents=True)
            (build_dir / "full_build_status.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.sft-full-build-status.v1",
                        "build_run_id": "full-build",
                        "train_ready": False,
                        "missing_markers": [
                            "manifest",
                            "rejection_report",
                            "sft_build_report",
                        ],
                        "next_action": "wait for or rerun full-dataset build before training",
                        "remote_dataset_dir": "/runs/full-build/sft-data",
                        "progress": {
                            "latest": {"processed_transitions": 100},
                            "restart_count": 0,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            result = runner.invoke(
                app,
                [
                    "sft",
                    "full-build-followup",
                    "--build-run-id",
                    "full-build",
                    "--train-run-id",
                    "full-train",
                    "--local-artifact-root",
                    str(root),
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["training_gate"]["status"], "closed")
        self.assertEqual(
            payload["training_gate"]["missing_markers"],
            ["manifest", "rejection_report", "sft_build_report"],
        )
        self.assertEqual(
            payload["training_gate"]["readiness_report_path"],
            str(build_dir / "full_build_status.json"),
        )
        self.assertEqual(payload["training_gate"]["readiness_source"], "status_report")

    def test_full_build_readiness_report_requires_completion_markers(self) -> None:
        from learn_nethack.full_build_readiness import (
            build_full_build_readiness_report,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "full-build").mkdir()
            (root / "full-build" / "manifest.json").write_text("{}", encoding="utf-8")
            report = build_full_build_readiness_report(
                build_run_id="full-build",
                local_artifact_root=root,
                remote_marker_status={
                    "train_jsonl": {"status": "present", "returncode": 0},
                    "manifest": {"status": "present", "returncode": 0},
                    "rejection_report": {
                        "status": "missing",
                        "returncode": 1,
                        "stderr": "No such file or directory",
                    },
                    "sft_build_report": {
                        "status": "missing",
                        "returncode": 1,
                        "stderr": "No such file or directory",
                    },
                },
            )

        self.assertFalse(report["train_ready"])
        self.assertEqual(
            report["missing_markers"], ["rejection_report", "sft_build_report"]
        )
        self.assertEqual(
            report["markers"]["train_jsonl"]["remote_path"],
            "/runs/full-build/sft-data/train.jsonl",
        )
        self.assertEqual(
            report["markers"]["manifest"]["remote_path"],
            "/runs/full-build/sft-data/manifest.json",
        )
        self.assertIn(
            "modal volume get --force learn-nethack-runs",
            report["markers"]["sft_build_report"]["modal_get_command"],
        )

    def test_full_build_readiness_report_summarizes_progress_ledger(self) -> None:
        from learn_nethack.full_build_readiness import (
            build_full_build_readiness_report,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_dir = root / "full-build"
            local_dir.mkdir()
            (local_dir / "sft_build_progress.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "schema_version": "learn-nethack.sft-build-progress.v1",
                                "processed_transitions": 1000,
                                "accepted_policy_rows": 500,
                                "accepted_next_frame_rows": 500,
                                "rejected_rows": 500,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "schema_version": "learn-nethack.sft-build-progress.v1",
                                "processed_transitions": 2000,
                                "accepted_policy_rows": 1200,
                                "accepted_next_frame_rows": 1200,
                                "rejected_rows": 800,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "schema_version": "learn-nethack.sft-build-progress.v1",
                                "processed_transitions": 500,
                                "accepted_policy_rows": 250,
                                "accepted_next_frame_rows": 250,
                                "rejected_rows": 250,
                            },
                            sort_keys=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_full_build_readiness_report(
                build_run_id="full-build",
                local_artifact_root=root,
            )

        self.assertTrue(report["progress"]["exists"])
        self.assertEqual(report["progress"]["event_count"], 3)
        self.assertEqual(report["progress"]["latest"]["processed_transitions"], 500)
        self.assertEqual(report["progress"]["max_processed_transitions"], 2000)
        self.assertEqual(report["progress"]["max_accepted_policy_rows"], 1200)
        self.assertEqual(report["progress"]["restart_count"], 1)

    def test_full_build_status_cli_writes_not_ready_report(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "status.json"
            result = runner.invoke(
                app,
                [
                    "sft",
                    "full-build-status",
                    "--build-run-id",
                    "full-build",
                    "--local-artifact-root",
                    str(root),
                    "--out",
                    str(out),
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            report = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(
            report["schema_version"],
            "learn-nethack.sft-full-build-status.v1",
        )
        self.assertFalse(report["train_ready"])
        self.assertEqual(
            report["missing_markers"],
            ["train_jsonl", "manifest", "rejection_report", "sft_build_report"],
        )

    def test_eval_status_report_requires_metrics_and_report_markers(self) -> None:
        from learn_nethack.eval_readiness import build_eval_status_report

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "eval-run"
            run_dir.mkdir()
            (run_dir / "sft_eval_contract.json").write_text("{}", encoding="utf-8")
            report = build_eval_status_report(
                eval_run_id="eval-run",
                local_artifact_root=root,
                remote_marker_status={
                    "contract": {"status": "present", "returncode": 0},
                    "metrics": {"status": "missing", "returncode": 1},
                    "report": {"status": "missing", "returncode": 1},
                    "progress": {"status": "missing", "returncode": 1},
                },
            )

        self.assertFalse(report["eval_ready"])
        self.assertEqual(report["missing_markers"], ["metrics", "report"])
        self.assertEqual(report["progress"]["exists"], False)
        self.assertEqual(report["next_action"], "wait for or rerun SFT eval")

    def test_eval_status_report_summarizes_progress_ledger(self) -> None:
        from learn_nethack.eval_readiness import build_eval_status_report

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "eval-run"
            run_dir.mkdir()
            (run_dir / "sft_eval_contract.json").write_text("{}", encoding="utf-8")
            (run_dir / "sft_eval_metrics.json").write_text("{}", encoding="utf-8")
            (run_dir / "sft_eval_report.json").write_text("{}", encoding="utf-8")
            (run_dir / "sft_eval_progress.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "phase": "next_frame_generate",
                                "evaluated_rows": 6,
                                "max_rows": 64,
                                "parse_valid": 6,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "phase": "next_frame_sequence_frame",
                                "horizon": 10,
                                "generated_frames": 37,
                                "max_windows": 64,
                                "parse_valid": 35,
                            },
                            sort_keys=True,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_eval_status_report(
                eval_run_id="eval-run",
                local_artifact_root=root,
            )

        self.assertTrue(report["eval_ready"])
        self.assertEqual(report["progress"]["event_count"], 2)
        self.assertEqual(
            report["progress"]["latest"]["phase"], "next_frame_sequence_frame"
        )
        self.assertEqual(report["progress"]["max_evaluated_rows"], 6)
        self.assertEqual(report["progress"]["max_generated_frames"], 37)

    def test_eval_status_cli_writes_report(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "eval_status.json"
            result = runner.invoke(
                app,
                [
                    "sft",
                    "eval-status",
                    "--eval-run-id",
                    "eval-run",
                    "--local-artifact-root",
                    str(root),
                    "--out",
                    str(out),
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            report = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(
            report["schema_version"],
            "learn-nethack.sft-eval-status.v1",
        )
        self.assertFalse(report["eval_ready"])
        self.assertEqual(report["missing_markers"], ["contract", "metrics", "report"])

    def test_modal_cloud_execution_context_requires_remote_function_ids(self) -> None:
        config = self._import_modal_config()

        with self.assertRaisesRegex(RuntimeError, "Modal readiness must run inside"):
            config.modal_cloud_execution_context(
                function_call_id=None,
                input_id=None,
                env={},
                hostname="local-host",
            )

        context = config.modal_cloud_execution_context(
            function_call_id="fc-123",
            input_id="in-456",
            env={"MODAL_TASK_ID": "ta-789"},
            hostname="modal-host",
        )

        self.assertEqual(context["backend"], "modal_cloud")
        self.assertEqual(context["function_call_id"], "fc-123")
        self.assertEqual(context["input_id"], "in-456")
        self.assertEqual(context["task_id"], "ta-789")
        self.assertEqual(context["hostname"], "modal-host")

    def test_modal_train_exposes_readiness_entrypoint_for_static_checks(self) -> None:
        try:
            modal_train = importlib.import_module("learn_nethack.modal_train")
        except ModuleNotFoundError as exc:
            self.fail(f"learn_nethack.modal_train must exist: {exc}")

        self.assertEqual(modal_train.MODAL_APP_NAME, "learn-nethack-gemma")
        self.assertTrue(callable(modal_train.readiness))
        self.assertTrue(callable(modal_train.local_readiness_report))
        self.assertTrue(callable(modal_train.sft_build))
        self.assertTrue(callable(modal_train.sft_train))
        self.assertTrue(callable(modal_train.sft_train_existing))
        self.assertTrue(callable(modal_train.sft_eval))
        self.assertTrue(callable(modal_train.sft_compare))
        self.assertTrue(callable(modal_train.watch_compare))
        self.assertTrue(callable(modal_train.watch_compare_sweep))
        self.assertTrue(callable(modal_train.extract_nld_shard))
        self.assertTrue(callable(modal_train.local_sft_build_contract))
        self.assertTrue(callable(modal_train.local_sft_train_contract))
        self.assertTrue(callable(modal_train.local_sft_train_existing_contract))
        self.assertTrue(callable(modal_train.local_watch_compare_contract))
        self.assertTrue(callable(modal_train.local_watch_compare_sweep_contract))

    def test_extract_nld_shard_local_fallback_writes_report(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "ana").mkdir()
            (source / "ana" / "a.ttyrec.bz2").write_text("ana", encoding="utf-8")
            shard = Path(tmp) / "nld-nao-000001.tar"
            build_player_tar_shard(
                source_root=source,
                player_names=["ana"],
                shard_path=shard,
            )
            destination = Path(tmp) / "dest"
            report = Path(tmp) / "extract-report.json"

            result = modal_train.extract_nld_shard(
                shard=str(shard),
                destination=str(destination),
                report=str(report),
            )

            self.assertTrue(report.exists())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["extract"]["player_count"], 1)
            self.assertEqual(
                (destination / "ana" / "a.ttyrec.bz2").read_text(encoding="utf-8"),
                "ana",
            )

    def test_local_sft_train_contract_names_inputs_outputs_and_proof_steps(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_train_contract(
            run_id="full-sft",
            db="/datasets/nld/nld-aa-taster/ttyrecs.db",
            action_manifest="/datasets/action_manifest.json",
            nle_root="/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
            mode="single_frame",
            full_dataset=True,
            max_rows=1000,
        )

        self.assertEqual(
            contract["schema_version"], "learn-nethack.sft-train-contract.v1"
        )
        self.assertEqual(contract["run_id"], "full-sft")
        self.assertEqual(
            contract["dataset"]["db"], "/datasets/nld/nld-aa-taster/ttyrecs.db"
        )
        self.assertEqual(
            contract["dataset"]["nle_root"],
            "/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
        )
        self.assertTrue(contract["dataset"]["full_dataset"])
        self.assertEqual(contract["artifacts"]["adapter"], "/runs/full-sft/adapters")
        self.assertEqual(
            contract["proof_steps"],
            [
                "build_full_sft_rows",
                "train_adapter",
                "eval_baseline_policy_and_next_frame",
                "eval_trained_policy_and_next_frame",
                "compare_policy_and_next_frame",
            ],
        )

    def test_local_sft_train_contract_supports_archive_manifest_source(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_train_contract(
            run_id="full-archive-sft",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
            mode="single_frame",
            full_dataset=True,
            max_rows=1000,
        )

        self.assertEqual(contract["dataset"]["source"], "archive_manifest")
        self.assertIsNone(contract["dataset"]["db"])
        self.assertIsNone(contract["dataset"]["nle_root"])
        self.assertEqual(
            contract["dataset"]["archive_manifest"],
            "/datasets/nld-nao-archive.jsonl",
        )
        self.assertEqual(
            contract["proof_steps"][0],
            "build_full_sft_rows_from_archive_shards",
        )

    def test_local_sft_train_contract_supports_archive_pseudo_labels(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_train_contract(
            run_id="full-archive-pseudo-sft",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
            mode="single_frame",
            full_dataset=True,
            max_rows=1000,
            tasks="policy_action",
            label_source="pseudo_visible_player_delta",
        )

        self.assertEqual(contract["dataset"]["source"], "archive_manifest")
        self.assertEqual(
            contract["dataset"]["label_source"], "pseudo_visible_player_delta"
        )
        self.assertEqual(contract["dataset"]["tasks"], ["policy_action"])
        self.assertEqual(
            contract["proof_steps"][0],
            "build_pseudo_label_sft_rows_from_archive_shards",
        )

    def test_local_sft_train_existing_contract_names_dataset_and_adapter(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_train_existing_contract(
            run_id="full-archive-existing-train",
            dataset_dir="/runs/full-archive-build/sft-data",
            model_name="google/gemma-4-E4b-it",
            max_steps=250,
            training_objective="dynamics_only",
        )

        self.assertEqual(
            contract["schema_version"],
            "learn-nethack.sft-train-existing-contract.v1",
        )
        self.assertEqual(contract["dataset"]["source"], "existing_sft_jsonl")
        self.assertEqual(
            contract["dataset"]["train_file"],
            "/runs/full-archive-build/sft-data/train.jsonl",
        )
        self.assertEqual(
            contract["artifacts"]["adapter"],
            "/runs/full-archive-existing-train/adapters",
        )
        self.assertEqual(contract["training"]["max_steps"], 250)
        self.assertEqual(contract["training"]["max_steps_mode"], "explicit")
        self.assertEqual(contract["training"]["training_objective"], "dynamics_only")
        self.assertEqual(
            contract["proof_steps"],
            [
                "verify_existing_sft_rows",
                "train_adapter",
                "eval_baseline_policy_and_next_frame",
                "eval_trained_policy_and_next_frame",
                "compare_policy_and_next_frame",
            ],
        )

    def test_local_sft_train_existing_contract_marks_auto_full_pass_steps(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_train_existing_contract(
            run_id="full-archive-existing-train",
            dataset_dir="/runs/full-archive-build/sft-data",
            max_steps=0,
        )

        self.assertEqual(contract["training"]["max_steps"], 0)
        self.assertEqual(
            contract["training"]["max_steps_mode"],
            "auto_one_full_pass",
        )

    def test_existing_sft_dataset_summary_requires_completion_markers(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "manifest"):
                modal_train._existing_sft_dataset_summary(dataset_dir)

            (dataset_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.sft-manifest.v1",
                        "accepted_policy_rows": 1,
                        "accepted_next_frame_rows": 1,
                        "rejected_rows": 0,
                        "tasks": ["policy_action", "next_frame"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "rejection report"):
                modal_train._existing_sft_dataset_summary(dataset_dir)

    def test_existing_sft_dataset_summary_reads_completed_manifest(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
            (dataset_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.sft-manifest.v1",
                        "accepted_policy_rows": 2,
                        "accepted_next_frame_rows": 2,
                        "rejected_rows": 1,
                        "tasks": ["policy_action", "next_frame"],
                    }
                ),
                encoding="utf-8",
            )
            (dataset_dir / "rejection_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.sft-rejections.v1",
                        "total_rejected": 1,
                    }
                ),
                encoding="utf-8",
            )

            summary = modal_train._existing_sft_dataset_summary(dataset_dir)

        self.assertEqual(summary["manifest"]["accepted_policy_rows"], 2)
        self.assertEqual(summary["rejection_report"]["total_rejected"], 1)
        self.assertEqual(
            summary["manifest_path"],
            str(dataset_dir / "manifest.json"),
        )

    def test_modal_sft_build_progress_logger_appends_durable_jsonl(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")
        from learn_nethack.sft_build import SftBuildProgress

        with TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "reports" / "sft_build_progress.jsonl"
            callback = modal_train._modal_sft_build_progress_logger(
                run_id="full-build",
                label_source="pseudo_visible_player_delta",
                tasks=("policy_action", "next_frame"),
                progress_path=progress_path,
            )
            callback(
                SftBuildProgress(
                    processed_transitions=2000,
                    accepted_policy_rows=1000,
                    accepted_next_frame_rows=1000,
                    rejected_rows=1000,
                    reason="accepted",
                    last_gameid=42,
                    last_step=17,
                )
            )
            event = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertEqual(event["schema_version"], "learn-nethack.sft-build-progress.v1")
        self.assertEqual(event["run_id"], "full-build")
        self.assertEqual(event["tasks"], ["policy_action", "next_frame"])
        self.assertEqual(event["processed_transitions"], 2000)

    def test_modal_sft_eval_progress_logger_appends_durable_jsonl(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        with TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "reports" / "sft_eval_progress.jsonl"
            callback = modal_train._modal_sft_eval_progress_logger(
                run_id="baseline-eval",
                label_source="pseudo_visible_player_delta",
                tasks=("policy_action", "next_frame"),
                progress_path=progress_path,
            )
            callback(
                {
                    "phase": "next_frame_sequence_frame",
                    "horizon": 10,
                    "window_index": 3,
                    "generated_frames": 37,
                    "max_windows": 64,
                    "parse_valid": 35,
                }
            )
            event = json.loads(progress_path.read_text(encoding="utf-8"))

        self.assertEqual(event["schema_version"], "learn-nethack.sft-eval-progress.v1")
        self.assertEqual(event["run_id"], "baseline-eval")
        self.assertEqual(event["tasks"], ["policy_action", "next_frame"])
        self.assertEqual(event["phase"], "next_frame_sequence_frame")
        self.assertEqual(event["generated_frames"], 37)

    def test_modal_sft_build_progress_logger_commits_runs_mount(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        self.assertEqual(
            modal_train._modal_volume_mount_for_path(
                Path("/runs/full-build/reports/sft_build_progress.jsonl")
            ),
            "/runs",
        )

    def test_local_sft_build_contract_supports_archive_pseudo_label_sizing(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_build_contract(
            run_id="full-archive-pseudo-sizing",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
            mode="single_frame",
            full_dataset=True,
            max_rows=1000,
            tasks="policy_action",
            label_source="pseudo_visible_player_delta",
        )

        self.assertEqual(
            contract["schema_version"], "learn-nethack.sft-build-contract.v1"
        )
        self.assertEqual(contract["dataset"]["source"], "archive_manifest")
        self.assertEqual(contract["dataset"]["max_rows"], None)
        self.assertEqual(
            contract["dataset"]["label_source"],
            "pseudo_visible_player_delta",
        )
        self.assertEqual(
            contract["proof_steps"],
            ["build_pseudo_label_sft_rows_from_archive_shards"],
        )

    def test_local_sft_build_contract_can_select_one_archive_shard(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_build_contract(
            run_id="full-archive-shard-000001",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
            archive_shard_index=1,
            mode="feedback_context_6",
            full_dataset=True,
            max_rows=1000,
            tasks="policy_action,next_frame",
            label_source="pseudo_visible_player_delta",
        )

        self.assertEqual(contract["dataset"]["archive_shard_index"], 1)
        self.assertTrue(contract["dataset"]["full_dataset"])
        self.assertEqual(
            contract["proof_steps"],
            ["build_pseudo_label_sft_rows_from_archive_shard"],
        )

    def test_local_sft_merge_shards_contract_names_completed_shard_inputs(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_merge_shards_contract(
            run_id="full-archive-merged",
            shard_run_ids="full-archive-shard-000000,full-archive-shard-000001",
        )

        self.assertEqual(
            contract["schema_version"],
            "learn-nethack.sft-merge-shards-contract.v1",
        )
        self.assertEqual(
            contract["shard_run_ids"],
            ["full-archive-shard-000000", "full-archive-shard-000001"],
        )
        self.assertEqual(
            contract["shard_dataset_dirs"],
            [
                "/runs/full-archive-shard-000000/sft-data",
                "/runs/full-archive-shard-000001/sft-data",
            ],
        )
        self.assertEqual(
            contract["artifacts"]["sft_data"], "/runs/full-archive-merged/sft-data"
        )

    def test_local_sft_train_contract_allows_pseudo_next_frame_task(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_train_contract(
            run_id="pseudo-dynamics-sft",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
            tasks="policy_action,next_frame",
            label_source="pseudo_visible_player_delta",
        )

        self.assertEqual(contract["dataset"]["tasks"], ["policy_action", "next_frame"])
        self.assertEqual(
            contract["dataset"]["label_source"], "pseudo_visible_player_delta"
        )

    def test_local_sft_train_contract_allows_bounded_smoke_steps(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_train_contract(
            run_id="archive-sft-smoke",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
            max_rows=32,
            max_steps=3,
        )

        self.assertEqual(contract["training"]["max_steps"], 3)
        self.assertEqual(contract["dataset"]["max_rows"], 32)

    def test_local_sft_train_contract_rejects_missing_dataset_source(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        with self.assertRaisesRegex(ValueError, "dataset source"):
            modal_train.local_sft_train_contract(
                run_id="bad-sft",
                db=None,
                action_manifest="/datasets/action_manifest.json",
            )

    def test_local_sft_eval_contract_requires_policy_and_teacher_forced_frame_metrics(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_eval_contract(
            run_id="full-sft-baseline",
            db="/datasets/nld/nld-aa-taster/ttyrecs.db",
            action_manifest="/datasets/action_manifest.json",
            nle_root="/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
        )

        self.assertIn("exact_match_rate", contract["required_metrics"])
        self.assertEqual(
            contract["evaluation"]["next_frame_eval_mode"],
            "teacher_forced",
        )
        self.assertIn(
            "next_frame_teacher_forced_mean_nll",
            contract["required_metrics"],
        )
        self.assertIn(
            "next_frame_teacher_forced_token_accuracy",
            contract["required_metrics"],
        )
        self.assertNotIn("next_frame_char_accuracy", contract["required_metrics"])
        self.assertEqual(
            contract["artifacts"]["progress"],
            "/runs/full-sft-baseline/reports/sft_eval_progress.jsonl",
        )

    def test_local_sft_eval_contract_can_limit_eval_tasks_and_frame_tokens(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_eval_contract(
            run_id="policy-only",
            db="/datasets/nld/nld-aa-taster/ttyrecs.db",
            action_manifest="/datasets/action_manifest.json",
            nle_root="/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
            eval_tasks="policy_action",
            next_frame_max_new_tokens=128,
        )

        self.assertEqual(contract["evaluation"]["tasks"], ["policy_action"])
        self.assertEqual(contract["evaluation"]["next_frame_max_new_tokens"], 128)
        self.assertIn("exact_match_rate", contract["required_metrics"])
        self.assertNotIn("next_frame_char_accuracy", contract["required_metrics"])

    def test_local_sft_eval_contract_can_request_generated_frame_metrics(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_eval_contract(
            run_id="generated-next-frame",
            db="/datasets/nld/nld-aa-taster/ttyrecs.db",
            action_manifest="/datasets/action_manifest.json",
            nle_root="/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
            eval_tasks="next_frame",
            next_frame_eval_mode="generate",
        )

        self.assertEqual(contract["evaluation"]["tasks"], ["next_frame"])
        self.assertEqual(contract["evaluation"]["next_frame_eval_mode"], "generate")
        self.assertEqual(
            contract["evaluation"]["next_frame_sequence_horizons"],
            [1, 5, 10],
        )
        self.assertEqual(contract["evaluation"]["next_frame_generate_max_rows"], 64)
        self.assertEqual(contract["evaluation"]["next_frame_sequence_max_windows"], 64)
        self.assertIn("next_frame_char_accuracy", contract["required_metrics"])
        self.assertIn("next_frame_parse_valid_rate", contract["required_metrics"])
        self.assertIn(
            "next_5_frame_sequence_char_accuracy",
            contract["required_metrics"],
        )
        self.assertIn(
            "next_10_frame_sequence_window_count",
            contract["required_metrics"],
        )
        self.assertIn(
            "next_10_frame_sequence_frame_count",
            contract["required_metrics"],
        )
        self.assertNotIn("exact_match_rate", contract["required_metrics"])

    def test_local_sft_eval_contract_rejects_empty_sequence_eval(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        with self.assertRaisesRegex(
            ValueError,
            "next_frame_sequence_max_windows",
        ):
            modal_train.local_sft_eval_contract(
                run_id="bad-sequence-eval",
                db="/datasets/nld/nld-aa-taster/ttyrecs.db",
                action_manifest="/datasets/action_manifest.json",
                nle_root="/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
                eval_tasks="next_frame",
                next_frame_eval_mode="generate",
                next_frame_sequence_max_windows=0,
            )

        with self.assertRaisesRegex(
            ValueError,
            "next_frame_generate_max_rows",
        ):
            modal_train.local_sft_eval_contract(
                run_id="bad-generate-eval",
                db="/datasets/nld/nld-aa-taster/ttyrecs.db",
                action_manifest="/datasets/action_manifest.json",
                nle_root="/datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data",
                eval_tasks="next_frame",
                next_frame_eval_mode="generate",
                next_frame_generate_max_rows=0,
            )

    def test_local_sft_eval_contract_supports_archive_manifest_source(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_eval_contract(
            run_id="full-archive-sft-baseline",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
        )

        self.assertEqual(contract["dataset"]["source"], "archive_manifest")
        self.assertIsNone(contract["dataset"]["db"])
        self.assertEqual(
            contract["dataset"]["archive_manifest"],
            "/datasets/nld-nao-archive.jsonl",
        )

    def test_local_sft_eval_contract_supports_archive_pseudo_dynamics(
        self,
    ) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_sft_eval_contract(
            run_id="full-archive-pseudo-eval",
            db=None,
            action_manifest="/datasets/action_manifest.json",
            archive_manifest="/datasets/nld-nao-archive.jsonl",
            eval_tasks="policy_action,next_frame",
            label_source="pseudo_visible_player_delta",
        )

        self.assertEqual(contract["dataset"]["source"], "archive_manifest")
        self.assertEqual(
            contract["dataset"]["label_source"],
            "pseudo_visible_player_delta",
        )
        self.assertIn("exact_match_rate", contract["required_metrics"])
        self.assertIn(
            "next_frame_teacher_forced_mean_nll",
            contract["required_metrics"],
        )

    def test_local_watch_compare_contract_names_modal_artifacts(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_watch_compare_contract(
            run_id="watch-10",
            action_manifest="/datasets/action_manifest.json",
            env_id="NetHack-v0",
            model_name="google/gemma-4-E2b-it",
            max_steps=10,
        )

        self.assertEqual(
            contract["schema_version"], "learn-nethack.watch-compare-contract.v1"
        )
        self.assertEqual(contract["run_id"], "watch-10")
        self.assertEqual(contract["watch"]["env_id"], "NetHack-v0")
        self.assertEqual(contract["watch"]["max_steps"], 10)
        self.assertEqual(contract["watch"]["model_name"], "google/gemma-4-E2b-it")
        self.assertIsNone(contract["watch"]["current_checkpoint"])
        self.assertEqual(
            contract["artifacts"]["watch_dir"],
            "/watch/watch-10",
        )
        self.assertEqual(
            contract["artifacts"]["events"],
            "/watch/watch-10/events.jsonl",
        )
        self.assertEqual(
            contract["artifacts"]["report"],
            "/watch/watch-10/report.json",
        )

    def test_local_watch_compare_sweep_contract_names_modal_artifacts(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")

        contract = modal_train.local_watch_compare_sweep_contract(
            run_id="watch-sweep",
            action_manifest="/datasets/action_manifest.json",
            env_id="NetHack-v0",
            model_name="google/gemma-4-E2b-it",
            max_steps=10,
            seeds="101,202",
        )

        self.assertEqual(
            contract["schema_version"],
            "learn-nethack.watch-compare-sweep-contract.v1",
        )
        self.assertEqual(contract["run_id"], "watch-sweep")
        self.assertEqual(contract["watch"]["seeds"], [101, 202])
        self.assertEqual(contract["watch"]["env_id"], "NetHack-v0")
        self.assertEqual(contract["watch"]["max_steps"], 10)
        self.assertEqual(
            contract["artifacts"]["watch_dir"],
            "/watch/watch-sweep",
        )
        self.assertEqual(
            contract["artifacts"]["report"],
            "/watch/watch-sweep/sweep_report.json",
        )
        self.assertEqual(
            contract["artifacts"]["contract"],
            "/runs/watch-sweep/reports/watch_compare_sweep_contract.json",
        )

    def test_watch_sweep_artifact_entries_keep_seed_paths_unique(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (101, 202):
                seed_dir = root / f"seed-{seed}"
                seed_dir.mkdir()
                for filename in ("events.jsonl", "report.json"):
                    (seed_dir / filename).write_text("{}", encoding="utf-8")

            entries = modal_train._watch_sweep_artifact_file_entries(
                [
                    {
                        "seed": 101,
                        "events_path": str(root / "seed-101" / "events.jsonl"),
                        "report_path": str(root / "seed-101" / "report.json"),
                    },
                    {
                        "seed": 202,
                        "events_path": str(root / "seed-202" / "events.jsonl"),
                        "report_path": str(root / "seed-202" / "report.json"),
                    },
                ]
            )

        names = [name for _path, name in entries]
        self.assertEqual(
            names,
            [
                "seed-101/events.jsonl",
                "seed-101/report.json",
                "seed-202/events.jsonl",
                "seed-202/report.json",
            ],
        )
        self.assertEqual(len(names), len(set(names)))

    def test_decoded_batch_source_reads_archive_manifest(self) -> None:
        modal_train = importlib.import_module("learn_nethack.modal_train")
        with TemporaryDirectory() as tmp:
            manifest = _make_archive_fixture(Path(tmp))
            observed: dict[str, object] = {}

            def fake_batch_iterator(**kwargs: Any):
                observed["dataset_name"] = kwargs["dataset_name"]
                observed["gameids"] = kwargs["gameids"]
                with sqlite3.connect(kwargs["dbfilename"]) as conn:
                    observed["staged_root"] = conn.execute(
                        "select root from roots"
                    ).fetchone()[0]
                yield {"gameids": [1], "keypresses": [107], "tty_chars": [[[64]]]}

            source = modal_train._decoded_batch_source(
                db=None,
                nle_root=None,
                archive_manifest=str(manifest),
                batch_size=4,
                seq_length=8,
                batch_iterator=fake_batch_iterator,
            )
            batches = list(source.iter_batches([1]))

        self.assertEqual(source.dataset_name, "fixture-nld")
        self.assertEqual(source.gameids, [1])
        self.assertEqual(source.archive_shard_count, 1)
        self.assertEqual(len(batches), 1)
        self.assertEqual(observed["dataset_name"], "fixture-nld")
        self.assertEqual(observed["gameids"], [1])

    def test_gitignore_blocks_generated_modal_training_artifacts(self) -> None:
        gitignore_path = ROOT / ".gitignore"

        self.assertTrue(gitignore_path.exists(), ".gitignore must exist")

        ignored = gitignore_path.read_text()

        for pattern in (
            "artifacts/",
            ".modal/",
            ".wandb/",
            ".uv-cache/",
            "wandb/",
            "*.ttyrec",
            "*.safetensors",
            "*.ckpt",
            ".env",
        ):
            self.assertIn(pattern, ignored)


if __name__ == "__main__":
    unittest.main()
