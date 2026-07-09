from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from learn_nethack.status_dashboard import write_status_dashboard


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
                                "description": "fixture",
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
        self.assertIn("live_rollout_utility_v7", html)
        self.assertIn("Goal active", html)


if __name__ == "__main__":
    unittest.main()
