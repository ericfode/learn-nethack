from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from learn_nethack.status_dashboard import (
    _best_corrected_policy_report,
    _goal_status,
    write_status_dashboard,
)


class StatusDashboardTests(unittest.TestCase):
    def test_writes_dashboard_snapshot_and_html_with_demo_links(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            build_run_id = "build-run"
            eval_run_id = "baseline-eval"
            build_dir = artifacts / build_run_id
            eval_dir = artifacts / eval_run_id
            watch_dir = artifacts / "watch" / "demo-watch"
            build_dir.mkdir(parents=True)
            eval_dir.mkdir(parents=True)
            watch_dir.mkdir(parents=True)

            (build_dir / "full_build_status.json").write_text(
                json.dumps(
                    {
                        "build_run_id": build_run_id,
                        "train_ready": False,
                        "missing_markers": ["manifest"],
                        "next_action": "wait for build",
                        "progress": {
                            "latest": {
                                "processed_transitions": 100,
                                "accepted_policy_rows": 40,
                                "accepted_next_frame_rows": 40,
                                "rejected_rows": 60,
                            },
                            "restart_count": 1,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (eval_dir / "eval_status.json").write_text(
                json.dumps(
                    {
                        "eval_run_id": eval_run_id,
                        "eval_ready": False,
                        "missing_markers": ["metrics", "report"],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (artifacts / "modal-shards").mkdir()
            (
                artifacts / "modal-shards" / "nld-nao-shard-000001.db-report.json"
            ).write_text(
                json.dumps(
                    {"selected_game_count": 3, "selected_ttyrec_count": 5},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (watch_dir / "report.json").write_text(
                json.dumps(
                    {
                        "run_id": "demo-watch",
                        "rollout_metrics": {
                            "current": {
                                "fitness_objective_version": "live_rollout_utility_v7",
                                "cumulative_reward": 1.0,
                                "score_delta": 2.0,
                                "hp_damage_observed": 0.0,
                                "wall_message_rate": 0.1,
                                "action_repeat_rate": 0.2,
                            },
                            "baseline": {"cumulative_reward": 0.0},
                            "deltas": {"cumulative_reward": 1.0},
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (watch_dir / "index.html").write_text("<html>demo</html>", encoding="utf-8")
            (artifacts / "score").mkdir()
            (artifacts / "score" / "score_to_beat.json").write_text(
                json.dumps(
                    {
                        "verdict": "mixed",
                        "metrics": {
                            "exact_match_rate": {
                                "baseline": 0.1,
                                "trained": 0.2,
                                "delta": 0.1,
                                "improved": True,
                            },
                            "next_1_frame_sequence_char_accuracy": {
                                "baseline": 0.3,
                                "trained": 0.4,
                                "delta": 0.1,
                                "improved": True,
                            },
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            (artifacts / "proof").mkdir()
            (artifacts / "proof" / "training_proof_gate.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.training-proof-gate.v1",
                        "passed": False,
                        "verdict": "failed",
                        "requirements": [
                            {
                                "name": "watch_score_or_depth_progress",
                                "status": "failed",
                                "reason": "needs progress",
                            }
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            local_watch = artifacts / "local-proof" / "watch"
            local_watch.mkdir(parents=True)
            (artifacts / "local-proof" / "report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.local-world-model-proof.v1",
                        "run_id": "local-proof",
                    }
                ),
                encoding="utf-8",
            )
            (local_watch / "index.html").write_text(
                "<html>world model</html>",
                encoding="utf-8",
            )
            summary = {
                key: {"mean": value, "sample_std": 0.01}
                for key, value in {
                    "one_step_changed_f1": 0.6,
                    "next_1_changed_f1": 0.5,
                    "next_5_changed_f1": 0.4,
                    "next_10_changed_f1": 0.3,
                    "action_mrr": 0.7,
                    "one_step_char_accuracy": 0.9,
                }.items()
            }
            (artifacts / "local-world-model-aggregate.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.local-world-model-aggregate.v1",
                        "verdict": "not_supported",
                        "run_count": 3,
                        "supported_run_count": 0,
                        "matched_parameter_count": 123,
                        "variants": {
                            "deterministic": summary,
                            "diffusion": summary,
                        },
                        "diffusion_minus_deterministic": {
                            key: {"mean": -0.1} for key in summary
                        },
                        "failure_summary": {
                            "action_mrr_nonpositive_delta_runs": 3,
                            "diffusion_next_10_f1_range": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = write_status_dashboard(
                repo_root=root,
                out_dir=artifacts / "status-dashboard",
                build_run_id=build_run_id,
                baseline_eval_run_id=eval_run_id,
                modal_runner=lambda command: {
                    "returncode": 0,
                    "stdout": json.dumps(
                        [
                            {
                                "app_id": "ap-test",
                                "description": "learn-nethack-gemma",
                                "state": "ephemeral (detached)",
                                "tasks": "1",
                            }
                        ]
                    ),
                    "stderr": "",
                    "command": " ".join(command),
                },
            )

            snapshot = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
            html = result.html_path.read_text(encoding="utf-8")

        self.assertFalse(snapshot["full_build"]["train_ready"])
        self.assertEqual(snapshot["shards"]["total_games"], 3)
        self.assertEqual(snapshot["modal_apps"]["apps"][0]["app_id"], "ap-test")
        self.assertIn("ap-test", html)
        self.assertIn("demo-watch", html)
        self.assertEqual(snapshot["world_model_proof"]["verdict"], "not_supported")
        self.assertIn("Local Decoder Proof", html)
        self.assertIn("World model: local-proof", html)
        self.assertIn("live_rollout_utility_v7", html)
        self.assertEqual(
            snapshot["goal_status"]["build_activity"],
            "withheld_by_proof_gate",
        )
        self.assertIn("Full-corpus scaling is withheld", html)

    def test_goal_status_marks_incomplete_build_stalled_without_matching_task(
        self,
    ) -> None:
        status = _goal_status(
            full_build={"build_run_id": "build-run", "train_ready": False},
            baseline_eval={"eval_ready": True},
            proof_gates=[],
            modal_apps={
                "status": "ok",
                "apps": [
                    {
                        "description": "unrelated-deployment",
                        "state": "deployed",
                        "tasks": "0",
                    }
                ],
            },
        )

        self.assertEqual(status["build_activity"], "stalled")
        self.assertEqual(status["label"], "Goal blocked: full build stalled")

    def test_goal_status_prioritizes_latest_live_proof_failure(self) -> None:
        status = _goal_status(
            full_build={"train_ready": False},
            baseline_eval={"eval_ready": False},
            proof_gates=[
                {
                    "passed": False,
                    "failed": [
                        {"name": "watch_current_action_repeat_rate_ceiling"},
                        {"name": "watch_current_score_or_depth_progress"},
                    ],
                }
            ],
            modal_apps={"status": "ok", "apps": []},
        )

        self.assertEqual(
            status["label"],
            "Corrected 20k rejected: live control failed",
        )
        self.assertEqual(status["build_activity"], "withheld_by_proof_gate")

    def test_best_policy_report_excludes_higher_historical_pseudo_result(
        self,
    ) -> None:
        report = _best_corrected_policy_report(
            [
                {
                    "run": "historical-pseudo-20k",
                    "exact_match_rate": {"trained": 0.9},
                },
                {
                    "run": "corrected20k-single-policy",
                    "exact_match_rate": {"trained": 0.6},
                },
            ]
        )

        self.assertEqual(report["run"], "corrected20k-single-policy")


if __name__ == "__main__":
    unittest.main()
