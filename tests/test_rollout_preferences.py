from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from typer.testing import CliRunner

from learn_nethack.cli import app
from learn_nethack.rollout_preferences import (
    build_policy_action_preference_rows,
    transition_utility_breakdown,
    write_rollout_preference_jsonl,
)


def _watch_report() -> dict:
    return {
        "schema_version": "learn-nethack.compare-watch-report.v1",
        "run_id": "watch-pref-fixture",
        "action_manifest": {
            "env_id": "NetHackChallenge-v0",
            "valid_action_ids": [0, 1],
        },
        "rollout_metrics": {
            "current": {
                "fitness_objective_version": "live_rollout_utility_v2",
                "fitness_score": -1.0,
            },
            "baseline": {
                "fitness_objective_version": "live_rollout_utility_v2",
                "fitness_score": 1.0,
            },
            "deltas": {"fitness_score": -2.0},
        },
        "events": [
            {
                "step": 0,
                "current": {
                    "policy_observation_text": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_terminal_frame": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_score": 0,
                    "prompt_hp": 12,
                    "prompt_depth": 1,
                    "action_id": 1,
                    "reward": 0.0,
                    "score": 0,
                    "hp": 12,
                    "depth": 1,
                    "message": "It's a wall.",
                    "game_time_advanced": True,
                },
                "baseline": {
                    "policy_observation_text": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_terminal_frame": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_score": 0,
                    "prompt_hp": 12,
                    "prompt_depth": 1,
                    "action_id": 0,
                    "reward": 1.0,
                    "score": 3,
                    "hp": 12,
                    "depth": 1,
                    "message": "You move.",
                    "game_time_advanced": True,
                },
            },
            {
                "step": 1,
                "current": {
                    "policy_observation_text": "MAP:\n@.\nMESSAGE:\nwall\n",
                    "action_id": 1,
                },
                "baseline": {
                    "policy_observation_text": "MAP:\n.@\nMESSAGE:\nmove\n",
                    "action_id": 0,
                },
            },
        ],
    }


def _realistic_prompt_frame(message: str) -> str:
    return (
        "MAP:\n"
        f"{message}\n\n"
        "           ---+--------\n"
        "           |........f.|\n"
        "           |(.........|\n"
        "           |.<.......@|\n"
        "           ------------\n"
        "MESSAGE:\n"
        f"{message}\n"
    )


class RolloutPreferenceTests(unittest.TestCase):
    def test_builds_same_prompt_policy_action_preference_rows(self) -> None:
        rows, report = build_policy_action_preference_rows(_watch_report())

        self.assertEqual(
            report["schema_version"], "learn-nethack.rollout-preference-build-report.v1"
        )
        self.assertEqual(report["row_count"], 1)
        self.assertEqual(report["skipped_counts"]["divergent_prompt"], 1)
        row = rows[0]
        self.assertEqual(
            row["schema_version"], "learn-nethack.policy-action-preference-row.v1"
        )
        self.assertEqual(row["task"], "policy_action_preference")
        self.assertIn("Allowed action_ids: [0, 1]", row["messages"][1]["content"])
        self.assertEqual(row["chosen"]["content"], '{"action_id": 0}')
        self.assertEqual(row["rejected"]["content"], '{"action_id": 1}')
        self.assertEqual(row["metadata"]["chosen_side"], "baseline")
        self.assertGreater(row["metadata"]["utility_margin"], 0.0)

    def test_skips_less_bad_preferences_without_positive_chosen_signal(self) -> None:
        report = _watch_report()
        event = report["events"][0]
        event["current"].update(
            {
                "action_id": 1,
                "message": "It's a wall.",
                "prompt_terminal_frame": _realistic_prompt_frame("It's a wall."),
                "terminal_frame": _realistic_prompt_frame("It's a wall."),
                "reward": 0.0,
                "score": 0,
                "hp": 12,
                "depth": 1,
                "game_time_advanced": True,
            }
        )
        event["baseline"].update(
            {
                "action_id": 0,
                "message": "What do you want to eat? [ghij or ?*]",
                "prompt_terminal_frame": _realistic_prompt_frame("It's a wall."),
                "terminal_frame": _realistic_prompt_frame(
                    "What do you want to eat? [ghij or ?*]"
                ),
                "reward": 0.0,
                "score": 0,
                "hp": 12,
                "depth": 1,
                "game_time_advanced": False,
            }
        )

        rows, build_report = build_policy_action_preference_rows(report)

        self.assertEqual(rows, [])
        self.assertEqual(build_report["skipped_counts"]["low_quality_preference"], 1)

    def test_transition_utility_detects_frame_only_menu_as_bad_signal(self) -> None:
        breakdown = transition_utility_breakdown(
            {
                "prompt_score": 0,
                "prompt_hp": 12,
                "prompt_depth": 1,
                "action_id": 30,
                "reward": 0.0,
                "score": 0,
                "hp": 12,
                "depth": 1,
                "message": "",
                "terminal_frame": (
                    "MAP:\n"
                    " Extended Commands List\n"
                    " eat                eat something\n"
                    " (1 of 5)\n"
                    "MESSAGE:\n<missing>\n"
                ),
                "game_time_advanced": False,
            }
        )

        self.assertTrue(breakdown.bad_signal)
        self.assertFalse(breakdown.positive_signal)
        self.assertLess(breakdown.value, 0.0)

    def test_transition_utility_does_not_treat_dirty_progress_as_positive(
        self,
    ) -> None:
        breakdown = transition_utility_breakdown(
            {
                "prompt_score": 0,
                "prompt_hp": 12,
                "prompt_depth": 1,
                "action_id": 30,
                "reward": 1.0,
                "score": 3,
                "hp": 12,
                "depth": 1,
                "message": "You cannot move there.",
                "terminal_frame": "MAP:\n@.\nMESSAGE:\nYou cannot move there.\n",
                "game_time_advanced": False,
            }
        )

        self.assertFalse(breakdown.positive_signal)
        self.assertTrue(breakdown.bad_signal)
        self.assertEqual(breakdown.components["live_progress_event_bonus"], 0.0)
        self.assertEqual(
            breakdown.components["dirty_live_progress_event_penalty"],
            -1.0,
        )
        self.assertEqual(breakdown.components["zero_progress_episode_penalty"], 0.0)

    def test_builds_preference_when_chosen_avoids_rejected_wall(self) -> None:
        report = _watch_report()
        report["events"] = [
            {
                "step": 0,
                "current": {
                    "policy_observation_text": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_terminal_frame": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_score": 0,
                    "prompt_hp": 12,
                    "prompt_depth": 1,
                    "action_id": 1,
                    "reward": 0.0,
                    "score": 0,
                    "hp": 12,
                    "depth": 1,
                    "message": "You move.",
                    "terminal_frame": "MAP:\n.@\nMESSAGE:\nYou move.\n",
                    "game_time_advanced": True,
                },
                "baseline": {
                    "policy_observation_text": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_terminal_frame": "MAP:\n@.\nMESSAGE:\n",
                    "prompt_score": 0,
                    "prompt_hp": 12,
                    "prompt_depth": 1,
                    "action_id": 0,
                    "reward": 0.0,
                    "score": 0,
                    "hp": 12,
                    "depth": 1,
                    "message": "It's a wall.",
                    "terminal_frame": "MAP:\n@.\nMESSAGE:\nIt's a wall.\n",
                    "game_time_advanced": True,
                },
            }
        ]

        rows, build_report = build_policy_action_preference_rows(report)

        self.assertEqual(build_report["row_count"], 1)
        self.assertEqual(rows[0]["chosen"]["content"], '{"action_id": 1}')
        self.assertFalse(rows[0]["metadata"]["chosen_positive_signal"])
        self.assertFalse(rows[0]["metadata"]["chosen_bad_signal"])
        self.assertTrue(rows[0]["metadata"]["rejected_bad_signal"])
        self.assertLess(rows[0]["metadata"]["chosen_utility"], 0.0)
        self.assertGreater(
            rows[0]["metadata"]["chosen_utility"],
            rows[0]["metadata"]["rejected_utility"],
        )

    def test_skips_preference_when_chosen_only_has_visible_novelty(self) -> None:
        report = _watch_report()
        report["events"] = [
            {
                "step": 0,
                "current": {
                    "policy_observation_text": "MAP:\n@..\nMESSAGE:\n",
                    "prompt_terminal_frame": "MAP:\n@..\nMESSAGE:\n",
                    "prompt_score": 0,
                    "prompt_hp": 12,
                    "prompt_depth": 1,
                    "action_id": 1,
                    "reward": 0.0,
                    "score": 0,
                    "hp": 12,
                    "depth": 1,
                    "message": "You move.",
                    "terminal_frame": "MAP:\n..@\nMESSAGE:\nYou move.\n",
                    "game_time_advanced": True,
                },
                "baseline": {
                    "policy_observation_text": "MAP:\n@..\nMESSAGE:\n",
                    "prompt_terminal_frame": "MAP:\n@..\nMESSAGE:\n",
                    "prompt_score": 0,
                    "prompt_hp": 12,
                    "prompt_depth": 1,
                    "action_id": 0,
                    "reward": 0.0,
                    "score": 0,
                    "hp": 12,
                    "depth": 1,
                    "message": "",
                    "terminal_frame": "MAP:\n@..\nMESSAGE:\n",
                    "game_time_advanced": True,
                },
            }
        ]

        rows, build_report = build_policy_action_preference_rows(report)

        self.assertEqual(rows, [])
        self.assertEqual(build_report["skipped_counts"]["utility_tie"], 1)

    def test_transition_utility_taxes_visible_novelty_without_live_progress(
        self,
    ) -> None:
        breakdown = transition_utility_breakdown(
            {
                "prompt_terminal_frame": "MAP:\n@..\nMESSAGE:\n",
                "prompt_score": 0,
                "prompt_hp": 12,
                "prompt_depth": 1,
                "action_id": 1,
                "reward": 0.0,
                "score": 0,
                "hp": 12,
                "depth": 1,
                "message": "You move.",
                "terminal_frame": "MAP:\n..@\nMESSAGE:\nYou move.\n",
                "game_time_advanced": True,
            }
        )

        self.assertFalse(breakdown.positive_signal)
        self.assertEqual(
            breakdown.components["zero_progress_episode_penalty"],
            -3.0,
        )
        self.assertEqual(breakdown.components["visible_map_novelty_bonus"], 0.0)
        self.assertEqual(breakdown.components["meaningful_event_bonus"], 0.0)
        self.assertLess(breakdown.value, 0.0)

    def test_writes_preference_jsonl_and_build_report(self) -> None:
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "preferences.jsonl"
            report_path = Path(tmp) / "preference_report.json"

            report = write_rollout_preference_jsonl(
                watch_report=_watch_report(),
                out_path=out,
                report_path=report_path,
            )

            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            written_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["row_count"], 1)
        self.assertEqual(written_report["row_count"], 1)
        self.assertEqual(rows[0]["chosen"]["content"], '{"action_id": 0}')

    def test_cli_builds_watch_preference_rows(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            watch_report = root / "watch_report.json"
            out = root / "preferences.jsonl"
            report_path = root / "preference_report.json"
            watch_report.write_text(
                json.dumps(_watch_report(), sort_keys=True),
                encoding="utf-8",
            )

            result = runner.invoke(
                app,
                [
                    "watch",
                    "build-preferences",
                    "--watch-report",
                    str(watch_report),
                    "--out",
                    str(out),
                    "--report",
                    str(report_path),
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            written_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(written_report["row_count"], 1)


if __name__ == "__main__":
    unittest.main()
