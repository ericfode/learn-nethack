from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from learn_nethack.action_manifest import ActionEntry, ActionManifest
from learn_nethack.cli import app
from learn_nethack.dynamics_play import (
    DynamicsModelSpec,
    build_dynamics_messages,
    parse_next_frame_response,
    read_initial_frame,
    read_ground_truth_frames_from_next_frame_rows,
    run_scripted_dynamics_session,
    validate_rendered_nethack_frame,
)


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
            ActionEntry(
                action_id=1,
                nle_action_name="CompassDirection.S",
                raw_key_code=106,
                key_label="j",
            ),
        ),
    )


class ScriptedDynamicsModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate_next_frame_json(
        self,
        *,
        observation_text: str,
        action_id: int,
        history: list[tuple[str, int]],
    ) -> str:
        del history
        self.calls.append((observation_text, action_id))
        step = len(self.calls)
        return json.dumps(
            {
                "next_frame": "\n".join(
                    [
                        "MAP:",
                        f".@{step}",
                        "MESSAGE:",
                        f"predicted action {action_id}",
                        "BLSTATS:",
                        "[1, 2, 3]",
                        "INVENTORY:",
                        "<empty>",
                    ]
                )
            },
            sort_keys=True,
        )


def _frame(*, message: str, blstats_value: int = 0) -> str:
    return "\n".join(
        [
            "MAP:",
            ".@.",
            "MESSAGE:",
            message,
            "BLSTATS:",
            json.dumps([blstats_value] * 27),
            "INVENTORY:",
            "<missing>",
        ]
    )


class DynamicsPlayTests(unittest.TestCase):
    def test_dynamics_prompt_uses_next_frame_contract(self) -> None:
        messages = build_dynamics_messages(
            observation_text="MAP:\n@.\nMESSAGE:\nhello",
            action_id=1,
            history=[],
        )

        self.assertEqual(
            messages[0]["content"],
            "You predict NetHack transition dynamics from NLE traces. "
            "Return only the next rendered observation frame text. "
            "Begin with MAP: and include MESSAGE:, BLSTATS:, and INVENTORY: sections.",
        )
        self.assertIn('Action taken: {"action_id": 1}', messages[1]["content"])
        self.assertIn("Current observation:\nMAP:\n@.", messages[1]["content"])

    def test_parse_next_frame_response_requires_exact_json_object(self) -> None:
        parsed = parse_next_frame_response('{"next_frame": "MAP:\\n@."}')

        self.assertEqual(parsed, "MAP:\n@.")

        raw_parsed = parse_next_frame_response("MAP:\n@.\nMESSAGE:\nhello")

        self.assertEqual(raw_parsed, "MAP:\n@.\nMESSAGE:\nhello")

        with self.assertRaisesRegex(ValueError, "expected only next_frame"):
            parse_next_frame_response('{"next_frame": "MAP", "extra": 1}')

        with self.assertRaisesRegex(ValueError, "invalid next-frame JSON"):
            parse_next_frame_response('assistant: {"next_frame": "MAP"}')

    def test_validate_rendered_nethack_frame_checks_shape_and_ground_truth(
        self,
    ) -> None:
        validation = validate_rendered_nethack_frame(
            _frame(message="predicted"),
            ground_truth_frame=_frame(message="ground truth"),
        )

        self.assertTrue(validation["rendered_frame_parse_valid"])
        self.assertTrue(validation["nle_observation_shape_valid"])
        self.assertTrue(validation["ground_truth_available"])
        self.assertFalse(validation["ground_truth_exact_match"])
        self.assertTrue(validation["map_exact_match"])
        self.assertFalse(validation["message_exact_match"])
        self.assertGreater(validation["char_accuracy"], 0.0)
        self.assertLess(validation["char_accuracy"], 1.0)

    def test_read_initial_frame_from_next_frame_row(self) -> None:
        row = {
            "task": "next_frame",
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": 'Action taken: {"action_id": 1}\nCurrent observation:\nMAP:\n@.\nMESSAGE:\nhello',
                },
                {"role": "assistant", "content": '{"next_frame": "ignored"}'},
            ],
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            frame = read_initial_frame(initial_frame_path=None, initial_row_path=path)

        self.assertEqual(frame, "MAP:\n@.\nMESSAGE:\nhello")

    def test_read_ground_truth_frames_from_next_frame_rows(self) -> None:
        rows = [
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": json.dumps({"next_frame": "A"})},
                ],
            },
            {"task": "policy_action", "messages": []},
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "MAP:\nB"},
                ],
                "metadata": {"next_frame_response_format": "raw_frame"},
            },
        ]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            frames = read_ground_truth_frames_from_next_frame_rows(path, max_frames=1)

        self.assertEqual(frames, ["A"])

    def test_scripted_session_writes_events_report_and_viewer(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            model = ScriptedDynamicsModel()

            report = run_scripted_dynamics_session(
                run_id="dynamics-smoke",
                model_spec=DynamicsModelSpec(
                    model_name="google/gemma-4-E4b-it",
                    adapter_checkpoint="artifacts/runs/dynamics-adapter",
                ),
                predictor=model,
                initial_frame="MAP:\n@.\nMESSAGE:\nstart",
                action_ids=[1, 0],
                action_manifest=_manifest(),
                out_dir=out_dir,
            )

            events_path = out_dir / "events.jsonl"
            html_path = out_dir / "index.html"
            report_path = out_dir / "report.json"

            self.assertTrue(events_path.exists())
            self.assertTrue(html_path.exists())
            self.assertTrue(report_path.exists())
            self.assertEqual(report["event_count"], 2)
            self.assertEqual(report["status"], "completed")

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            events[0]["schema_version"], "learn-nethack.dynamics-play-event.v1"
        )
        self.assertEqual(events[0]["action_id"], 1)
        self.assertEqual(events[0]["action_label"], "j")
        self.assertEqual(events[0]["status"], "predicted")
        self.assertIn("predicted action 1", events[0]["predicted_frame"])
        self.assertIn("predicted action 1", events[1]["prompt_frame"])
        self.assertEqual(model.calls[1][1], 0)

    def test_scripted_session_can_render_ground_truth_and_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            run_scripted_dynamics_session(
                run_id="dynamics-ground-truth-smoke",
                model_spec=DynamicsModelSpec(
                    model_name="google/gemma-4-E4b-it",
                    adapter_checkpoint="artifacts/runs/dynamics-adapter",
                ),
                predictor=ScriptedDynamicsModel(),
                initial_frame="MAP:\n.@.\nMESSAGE:\nstart\nBLSTATS:\n"
                + json.dumps([0] * 27)
                + "\nINVENTORY:\n<missing>",
                action_ids=[1],
                ground_truth_frames=[_frame(message="ground truth")],
                action_manifest=_manifest(),
                out_dir=out_dir,
            )

            event = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
            html = (out_dir / "index.html").read_text(encoding="utf-8")

        self.assertEqual(event["ground_truth_frame"], _frame(message="ground truth"))
        self.assertTrue(event["validation"]["ground_truth_available"])
        self.assertFalse(event["validation"]["ground_truth_exact_match"])
        self.assertIn("Ground Truth Next Frame", html)
        self.assertIn("groundTruthFrame", html)
        self.assertIn("truth exact", html)

    def test_scripted_session_rejects_out_of_manifest_action(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "not in active action manifest"):
                run_scripted_dynamics_session(
                    run_id="dynamics-smoke",
                    model_spec=DynamicsModelSpec(
                        model_name="google/gemma-4-E4b-it",
                        adapter_checkpoint=None,
                    ),
                    predictor=ScriptedDynamicsModel(),
                    initial_frame="MAP:\n@.",
                    action_ids=[99],
                    action_manifest=_manifest(),
                    out_dir=Path(tmp),
                )

    def test_cli_writes_dynamics_play_contract_without_heavy_deps(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "action_manifest.json"
            out_dir = root / "play"
            _manifest().save(manifest_path)

            result = runner.invoke(
                app,
                [
                    "play",
                    "dynamics",
                    "--run-id",
                    "contract-smoke",
                    "--action-manifest",
                    str(manifest_path),
                    "--adapter-checkpoint",
                    "artifacts/runs/dynamics-adapter",
                    "--out",
                    str(out_dir),
                    "--dry-run-contract",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            contract = json.loads((out_dir / "dynamics_play_contract.json").read_text())

        self.assertEqual(
            contract["schema_version"], "learn-nethack.dynamics-play-contract.v1"
        )
        self.assertEqual(contract["run_id"], "contract-smoke")
        self.assertEqual(
            contract["model"]["adapter_checkpoint"], "artifacts/runs/dynamics-adapter"
        )
        self.assertEqual(contract["valid_action_ids"], [0, 1])


if __name__ == "__main__":
    unittest.main()
