from __future__ import annotations

import json
import sqlite3
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from learn_nethack.action_manifest import (
    ActionEntry,
    ActionManifest,
    build_action_manifest_from_nle_actions,
    load_action_manifest,
)
from learn_nethack.cli import app
from learn_nethack.cli import resolve_build_row_limit, resolve_split_row_limits
from learn_nethack.nld_decode import normalize_decoded_batch, normalize_frame_only_batch
from learn_nethack.nld_metadata import (
    classify_sft_label_readiness,
    inspect_nld_db,
    order_gameids_for_split_role_coverage,
    split_gameids,
)
from learn_nethack.observations import render_observation_text
from learn_nethack.pseudo_labels import infer_visible_movement_pseudo_label
import learn_nethack.sft_build as sft_build
from learn_nethack.sft_build import (
    SplitRowLimits,
    build_pseudo_label_sft_from_frame_batches,
    build_sft_from_decoded_batches,
    merge_sft_dataset_shards,
    write_pseudo_label_policy_dataset,
    write_sft_dataset,
)
from learn_nethack.sft_eval import (
    build_score_to_beat_report,
    build_training_proof_gate_report,
    compute_next_frame_metrics,
    compute_policy_metrics,
    evaluate_next_frame_rows_with_scorer,
    evaluate_next_frame_sequences_with_predictor,
    evaluate_next_frame_rows_with_predictor,
    evaluate_policy_rows_with_policy,
    summarize_next_frame_sequence_rows,
)
from learn_nethack.sft_rows import (
    HistoryBuffer,
    build_next_frame_row,
    build_pseudo_next_frame_row,
    build_pseudo_policy_action_row,
    build_policy_action_row,
    policy_feedback_from_outcome_observation,
)


def _make_fixture_db(path: Path, *, ttyrec_version: int = 3) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "create table roots (dataset_name text primary key, root text, ttyrec_version integer)"
        )
        conn.execute(
            "create table games (gameid integer primary key, role text, race text, align text, death text, points integer, turns integer)"
        )
        conn.execute(
            "create table ttyrecs (path text, part integer, size integer, mtime real, gameid integer)"
        )
        conn.execute(
            "insert into roots values ('fixture-nld', '/tmp/fixture-nld/nle_data', ?)",
            (ttyrec_version,),
        )
        for gameid in (1, 2, 3):
            conn.execute(
                "insert into games values (?, 'Sam', 'Hum', 'Law', 'quit', 10, 20)",
                (gameid,),
            )
            conn.execute(
                "insert into ttyrecs values (?, 0, 100, 0.0, ?)",
                (f"game-{gameid}.ttyrec3.bz2", gameid),
            )


def _passing_policy_and_next_frame_score_report() -> dict:
    baseline_metrics = {
        "parse_valid_rate": 1.0,
        "action_space_valid_rate": 1.0,
        "exact_match_rate": 0.30,
        "next_frame_parse_valid_rate": 1.0,
        "next_frame_teacher_forced_mean_nll": 1.0,
    }
    trained_metrics = {
        "parse_valid_rate": 1.0,
        "action_space_valid_rate": 1.0,
        "exact_match_rate": 0.31,
        "next_frame_parse_valid_rate": 1.0,
        "next_frame_teacher_forced_mean_nll": 0.9,
    }
    for horizon, baseline_accuracy, trained_accuracy in (
        (1, 0.50, 0.60),
        (5, 0.40, 0.50),
        (10, 0.30, 0.45),
    ):
        baseline_metrics.update(
            {
                f"next_{horizon}_frame_sequence_available_window_count": 2.0,
                f"next_{horizon}_frame_sequence_available_frame_count": float(
                    horizon * 2
                ),
                f"next_{horizon}_frame_sequence_window_count": 2.0,
                f"next_{horizon}_frame_sequence_frame_count": float(horizon * 2),
                f"next_{horizon}_frame_sequence_parse_valid_rate": 1.0,
                f"next_{horizon}_frame_sequence_char_accuracy": baseline_accuracy,
                f"next_{horizon}_frame_sequence_exact_match_rate": 0.0,
                f"next_{horizon}_frame_sequence_changed_map_cell_f1": baseline_accuracy,
                f"next_{horizon}_frame_sequence_player_coordinate_exact_rate": 0.5,
                f"next_{horizon}_frame_sequence_blstats_field_exact_rate": 0.5,
                f"next_{horizon}_frame_sequence_message_edit_similarity": 0.5,
            }
        )
        trained_metrics.update(
            {
                f"next_{horizon}_frame_sequence_available_window_count": 2.0,
                f"next_{horizon}_frame_sequence_available_frame_count": float(
                    horizon * 2
                ),
                f"next_{horizon}_frame_sequence_window_count": 2.0,
                f"next_{horizon}_frame_sequence_frame_count": float(horizon * 2),
                f"next_{horizon}_frame_sequence_parse_valid_rate": 1.0,
                f"next_{horizon}_frame_sequence_char_accuracy": trained_accuracy,
                f"next_{horizon}_frame_sequence_exact_match_rate": 0.0,
                f"next_{horizon}_frame_sequence_changed_map_cell_f1": trained_accuracy,
                f"next_{horizon}_frame_sequence_player_coordinate_exact_rate": 0.5,
                f"next_{horizon}_frame_sequence_blstats_field_exact_rate": 0.5,
                f"next_{horizon}_frame_sequence_message_edit_similarity": 0.5,
            }
        )
    return build_score_to_beat_report(
        baseline_metrics=baseline_metrics,
        trained_metrics=trained_metrics,
        baseline_run_id="base",
        trained_run_id="trained",
    )


def _clean_watch_side(**overrides: object) -> dict[str, object]:
    side = {
        "fitness_objective_version": "live_rollout_utility_v7",
        "fitness_score": 2.0,
        "cumulative_reward": 1.0,
        "score_delta": 1.0,
        "depth_max": 1,
        "depth_delta": 0.0,
        "hp_damage_observed": 0.0,
        "wall_message_rate": 0.0,
        "bad_message_rate": 0.0,
        "non_advancing_step_rate": 0.0,
        "action_repeat_rate": 0.40,
        "starvation_or_faint_count": 0.0,
        "menu_or_prompt_step_rate": 0.0,
        "stuck_menu_or_prompt_loop_count": 0.0,
        "dirty_live_progress_event_count": 0.0,
        "zero_progress_episode": 0.0,
    }
    side.update(overrides)
    return side


def _manifest() -> ActionManifest:
    return ActionManifest(
        env_id="NetHackChallenge-v0",
        entries=(
            ActionEntry(
                action_id=0, nle_action_name="NORTH", raw_key_code=107, key_label="k"
            ),
            ActionEntry(
                action_id=1, nle_action_name="SOUTH", raw_key_code=106, key_label="j"
            ),
        ),
    )


class AlwaysSouthPolicy:
    def score_actions(
        self,
        *,
        observation_text: str,
        valid_action_ids: list[int],
    ) -> dict[int, float]:
        self.last_observation_text = observation_text
        return {
            action_id: (1.0 if action_id == 1 else 0.0)
            for action_id in valid_action_ids
        }


class EchoNextFramePredictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_next_frame_json(
        self,
        *,
        observation_text: str,
        action_id: int,
        history: list[tuple[str, int]],
    ) -> str:
        self.calls.append(
            {
                "observation_text": observation_text,
                "action_id": action_id,
                "history": history,
            }
        )
        if action_id == 1:
            return json.dumps({"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"})
        return json.dumps({"next_frame": "MAP:\n@.\nMESSAGE:\nStayed"})


class FixedNextFrameScorer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def score_next_frame_response(
        self,
        *,
        observation_text: str,
        action_id: int,
        target_response: str,
        history: list[tuple[str, int]],
    ) -> dict[str, float]:
        self.calls.append(
            {
                "observation_text": observation_text,
                "action_id": action_id,
                "target_response": target_response,
                "history": history,
            }
        )
        if action_id == 1:
            return {
                "token_count": 4.0,
                "negative_log_likelihood": 2.0,
                "argmax_match_count": 3.0,
            }
        return {
            "token_count": 2.0,
            "negative_log_likelihood": 4.0,
            "argmax_match_count": 1.0,
        }


def _install_fake_nle_actions() -> tuple[
    types.ModuleType | None, types.ModuleType | None
]:
    fake_nle = types.ModuleType("nle")
    fake_nethack = types.ModuleType("nle.nethack")

    class FakeAction(int):
        def __new__(cls, value: int, name: str):
            item = int.__new__(cls, value)
            item.name = name
            return item

    fake_nethack.ACTIONS = (
        FakeAction(107, "N"),
        FakeAction(106, "S"),
        FakeAction(13, "MORE"),
    )
    fake_nle.nethack = fake_nethack
    original_nle = sys.modules.get("nle")
    original_nethack = sys.modules.get("nle.nethack")
    sys.modules["nle"] = fake_nle
    sys.modules["nle.nethack"] = fake_nethack
    return original_nle, original_nethack


def _restore_fake_nle_actions(
    original_nle: types.ModuleType | None,
    original_nethack: types.ModuleType | None,
) -> None:
    if original_nle is None:
        sys.modules.pop("nle", None)
    else:
        sys.modules["nle"] = original_nle
    if original_nethack is None:
        sys.modules.pop("nle.nethack", None)
    else:
        sys.modules["nle.nethack"] = original_nethack


def _decoded_transitions():
    batch = {
        "gameids": [1, 1, 2],
        "steps": [0, 1, 0],
        "keypresses": [107, 106, 999],
        "tty_chars": [
            [[64, 46], [46, 46]],
            [[46, 64], [46, 46]],
            [[64, 46], [35, 35]],
        ],
        "message": [[72, 105], [77, 111, 118, 101, 100], [66, 108, 111, 99, 107]],
        "blstats": [[1, 2, 3], [2, 2, 3], [9, 9, 9]],
    }
    return list(normalize_decoded_batch(batch))


class _TrackedArray:
    def __init__(
        self,
        value: object,
        *,
        tolist_paths: list[tuple[int, ...]],
        path: tuple[int, ...] = (),
    ) -> None:
        self._value = value
        self._tolist_paths = tolist_paths
        self._path = path

    def __getitem__(self, index: int) -> object:
        value = self._value[index]  # type: ignore[index]
        if isinstance(value, list):
            return _TrackedArray(
                value,
                tolist_paths=self._tolist_paths,
                path=(*self._path, index),
            )
        return value

    def tolist(self) -> object:
        self._tolist_paths.append(self._path)
        return self._value


class NldSftDataLoopTests(unittest.TestCase):
    def test_inspect_nld_db_reads_metadata_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ttyrecs.db"
            _make_fixture_db(db_path)

            report = inspect_nld_db(db_path)

        self.assertEqual(report.dataset_name, "fixture-nld")
        self.assertEqual(report.ttyrec_version, 3)
        self.assertEqual(report.game_count, 3)
        self.assertEqual(report.ttyrec_count, 3)

    def test_classify_sft_label_readiness_accepts_ttyrec3_keypress_batches(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ttyrecs.db"
            _make_fixture_db(db_path, ttyrec_version=3)
            report = classify_sft_label_readiness(
                inspect_nld_db(db_path),
                decoded_batch_keys={"done", "gameids", "keypresses", "tty_chars"},
            )

        self.assertTrue(report.policy_action_trainable)
        self.assertTrue(report.next_frame_trainable)
        self.assertEqual(report.status, "labelled")
        self.assertEqual(report.reason, "decoded_action_labels_available")

    def test_classify_sft_label_readiness_rejects_ttyrec1_frame_only_batches(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "ttyrecs.db"
            _make_fixture_db(db_path, ttyrec_version=1)
            report = classify_sft_label_readiness(
                inspect_nld_db(db_path),
                decoded_batch_keys={
                    "done",
                    "gameids",
                    "timestamps",
                    "tty_chars",
                    "tty_colors",
                    "tty_cursor",
                },
            )

        self.assertFalse(report.policy_action_trainable)
        self.assertFalse(report.next_frame_trainable)
        self.assertEqual(report.status, "frame_only")
        self.assertEqual(report.reason, "decoded_action_labels_missing")
        self.assertIn("keypresses", report.required_decoded_keys)

    def test_split_gameids_is_stable_and_episode_safe(self) -> None:
        splits = split_gameids([1, 2, 3, 4, 5], seed=20260615)

        all_ids = splits.train + splits.validation + splits.test

        self.assertEqual(sorted(all_ids), [1, 2, 3, 4, 5])
        self.assertFalse(set(splits.train) & set(splits.validation))
        self.assertFalse(set(splits.train) & set(splits.test))
        self.assertEqual(split_gameids([1, 2, 3, 4, 5], seed=20260615), splits)

    def test_split_role_order_frontloads_heldout_role_coverage(self) -> None:
        gameids = list(range(1, 601))
        metadata = {
            gameid: {"role": ("Arc", "Hea", "Wiz")[gameid % 3]} for gameid in gameids
        }
        splits = split_gameids(gameids, seed=17)
        split_by_gameid = {
            gameid: split
            for split, values in (
                ("train", splits.train),
                ("validation", splits.validation),
                ("test", splits.test),
            )
            for gameid in values
        }

        ordered = order_gameids_for_split_role_coverage(
            gameids,
            game_metadata_by_id=metadata,
            seed=17,
        )
        prefix_pairs = {
            (split_by_gameid[gameid], metadata[gameid]["role"])
            for gameid in ordered[:9]
        }

        self.assertEqual(sorted(ordered), gameids)
        self.assertEqual(len(ordered), len(set(ordered)))
        self.assertEqual(
            prefix_pairs,
            {
                (split, role)
                for split in ("train", "validation", "test")
                for role in ("Arc", "Hea", "Wiz")
            },
        )

    def test_action_manifest_maps_raw_key_to_action_id(self) -> None:
        manifest = _manifest()

        self.assertEqual(manifest.action_id_for_raw_key(107), 0)
        self.assertEqual(manifest.valid_action_ids(), [0, 1])

        with self.assertRaisesRegex(KeyError, "raw key code 999"):
            manifest.action_id_for_raw_key(999)

    def test_action_manifest_rejects_ambiguous_raw_key_code(self) -> None:
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=0,
                    nle_action_name="Command.SEESPELLS",
                    raw_key_code=43,
                    key_label="+",
                ),
                ActionEntry(
                    action_id=1,
                    nle_action_name="TextCharacters.PLUS",
                    raw_key_code=43,
                    key_label="+",
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "ambiguous raw key code 43"):
            manifest.action_id_for_raw_key(43)

    def test_build_action_manifest_from_nle_actions_uses_action_order(self) -> None:
        original_nle, original_nethack = _install_fake_nle_actions()
        try:
            manifest = build_action_manifest_from_nle_actions(env_id="FakeEnv-v0")
        finally:
            _restore_fake_nle_actions(original_nle, original_nethack)

        self.assertEqual(manifest.valid_action_ids(), [0, 1, 2])
        self.assertEqual(manifest.action_id_for_raw_key(107), 0)
        self.assertEqual(manifest.action_id_for_raw_key(13), 2)
        self.assertEqual(manifest.entries[0].nle_action_name, "FakeAction.N")
        self.assertEqual(manifest.entries[0].key_label, "k")
        self.assertEqual(manifest.entries[2].key_label, "keycode_13")

    def test_build_action_manifest_uses_requested_env_action_space_when_available(
        self,
    ) -> None:
        try:
            import gymnasium as gym
            import nle  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("local NLE is not installed") from exc

        manifest = build_action_manifest_from_nle_actions(env_id="NetHack-v0")
        env = gym.make("NetHack-v0")
        try:
            expected_action_count = len(env.unwrapped.actions)
        finally:
            env.close()

        self.assertEqual(len(manifest.entries), expected_action_count)
        self.assertEqual(len(manifest.entries), 86)

    def test_cli_writes_action_manifest(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "action_manifest.json"
            original_nle, original_nethack = _install_fake_nle_actions()
            try:
                result = runner.invoke(
                    app,
                    [
                        "data",
                        "write-action-manifest",
                        "--out",
                        str(out_path),
                        "--env-id",
                        "FakeEnv-v0",
                    ],
                )
            finally:
                _restore_fake_nle_actions(original_nle, original_nethack)

            self.assertEqual(result.exit_code, 0, result.output)
            manifest = load_action_manifest(out_path)

        self.assertEqual(manifest.valid_action_ids(), [0, 1, 2])
        self.assertEqual(manifest.action_id_for_raw_key(106), 1)

    def test_cli_compares_baseline_and_trained_metric_reports(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            trained = root / "trained.json"
            out = root / "score_to_beat.json"
            baseline.write_text(
                json.dumps({"exact_match_rate": 0.25}) + "\n",
                encoding="utf-8",
            )
            trained.write_text(
                json.dumps({"exact_match_rate": 0.40}) + "\n",
                encoding="utf-8",
            )

            result = runner.invoke(
                app,
                [
                    "sft",
                    "compare",
                    "--baseline",
                    str(baseline),
                    "--trained",
                    str(trained),
                    "--out",
                    str(out),
                    "--baseline-run-id",
                    "base",
                    "--trained-run-id",
                    "sft",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            report = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(report["verdict"], "improved")
        self.assertEqual(report["metrics"]["exact_match_rate"]["delta"], 0.15)

    def test_resolve_build_row_limit_requires_explicit_full_dataset(self) -> None:
        self.assertEqual(
            resolve_build_row_limit(max_rows=1000, full_dataset=False), 1000
        )
        self.assertIsNone(resolve_build_row_limit(max_rows=1000, full_dataset=True))

        with self.assertRaisesRegex(ValueError, "full dataset"):
            resolve_build_row_limit(max_rows=64, full_dataset=True)

    def test_resolve_split_row_limits_requires_all_three_positive(self) -> None:
        limits = resolve_split_row_limits(
            train_rows=20_000,
            validation_rows=2_000,
            test_rows=2_000,
            full_dataset=False,
        )

        self.assertEqual(
            limits,
            SplitRowLimits(train=20_000, validation=2_000, test=2_000),
        )
        self.assertIsNone(
            resolve_split_row_limits(
                train_rows=1_000,
                validation_rows=0,
                test_rows=0,
                full_dataset=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "all be positive"):
            resolve_split_row_limits(
                train_rows=20_000,
                validation_rows=2_000,
                test_rows=0,
                full_dataset=False,
            )

    def test_normalize_decoded_batch_pairs_same_episode_next_observation(self) -> None:
        transitions = _decoded_transitions()

        self.assertEqual(transitions[0].gameid, 1)
        self.assertEqual(transitions[0].raw_key_code, 107)
        self.assertEqual(
            transitions[0].next_observation["tty_chars"], [[46, 64], [46, 46]]
        )
        self.assertIsNone(transitions[1].next_observation)
        self.assertIsNone(transitions[2].next_observation)

    def test_normalize_decoded_batch_converts_selected_frame_not_entire_array(
        self,
    ) -> None:
        tolist_paths: list[tuple[int, ...]] = []
        tty_chars = _TrackedArray(
            [[[[64, 46]], [[46, 64]]]],
            tolist_paths=tolist_paths,
        )
        batch = {
            "gameids": [[1, 1]],
            "steps": [[0, 1]],
            "keypresses": [[107, 106]],
            "tty_chars": tty_chars,
        }

        transitions = list(normalize_decoded_batch(batch))

        self.assertEqual(transitions[0].observation["tty_chars"], [[64, 46]])
        self.assertNotIn((), tolist_paths)
        self.assertTrue(tolist_paths)
        self.assertTrue(all(len(path) == 2 for path in tolist_paths))

    def test_normalize_decoded_batch_reports_available_keys_when_actions_missing(
        self,
    ) -> None:
        batch = {
            "gameids": [1],
            "tty_chars": [[[64, 46]]],
        }

        with self.assertRaisesRegex(
            ValueError,
            "available keys: gameids, tty_chars",
        ):
            list(normalize_decoded_batch(batch))

    def test_normalize_frame_only_batch_pairs_observations_without_keypresses(
        self,
    ) -> None:
        batch = {
            "gameids": [1, 1, 2],
            "steps": [0, 1, 0],
            "tty_chars": [
                [[64, 46], [46, 46]],
                [[46, 64], [46, 46]],
                [[35, 64], [35, 35]],
            ],
        }

        transitions = list(normalize_frame_only_batch(batch))

        self.assertEqual(len(transitions), 3)
        self.assertEqual(transitions[0].gameid, 1)
        self.assertEqual(transitions[0].step, 0)
        self.assertIsNotNone(transitions[0].next_observation)
        self.assertEqual(
            transitions[0].next_observation["tty_chars"], [[46, 64], [46, 46]]
        )
        self.assertIsNone(transitions[1].next_observation)
        self.assertIsNone(transitions[2].next_observation)

    def test_infer_visible_movement_pseudo_label_from_player_delta(self) -> None:
        transition = list(
            normalize_frame_only_batch(
                {
                    "gameids": [1, 1],
                    "tty_chars": [
                        [[64, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                    ],
                }
            )
        )[0]
        manifest = ActionManifest(
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
                    nle_action_name="CompassDirection.E",
                    raw_key_code=108,
                    key_label="l",
                ),
                ActionEntry(
                    action_id=2,
                    nle_action_name="MiscDirection.WAIT",
                    raw_key_code=46,
                    key_label=".",
                ),
            ),
        )

        label = infer_visible_movement_pseudo_label(
            transition=transition,
            action_manifest=manifest,
        )

        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.action_id, 1)
        self.assertEqual(label.direction, "E")
        self.assertEqual(label.label_source, "pseudo_visible_player_delta")
        self.assertEqual(label.confidence, 1.0)

    def test_infer_visible_movement_pseudo_label_rejects_missing_player(self) -> None:
        transition = list(
            normalize_frame_only_batch(
                {
                    "gameids": [1, 1],
                    "tty_chars": [
                        [[46, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                    ],
                }
            )
        )[0]

        self.assertIsNone(
            infer_visible_movement_pseudo_label(
                transition=transition,
                action_manifest=_manifest(),
            )
        )

    def test_normalize_real_nle_sequence_batch_flattens_batch_and_time_axes(
        self,
    ) -> None:
        batch = {
            "gameids": [[7, 7, 8], [9, 9, 9]],
            "keypresses": [[107, 106, 999], [106, 107, 106]],
            "tty_chars": [
                [
                    [[64, 46], [46, 46]],
                    [[46, 64], [46, 46]],
                    [[35, 35], [64, 46]],
                ],
                [
                    [[64, 35], [46, 46]],
                    [[46, 35], [64, 46]],
                    [[46, 46], [64, 35]],
                ],
            ],
            "tty_colors": [
                [
                    [[7, 7], [7, 7]],
                    [[7, 7], [7, 7]],
                    [[7, 7], [7, 7]],
                ],
                [
                    [[7, 7], [7, 7]],
                    [[7, 7], [7, 7]],
                    [[7, 7], [7, 7]],
                ],
            ],
            "tty_cursor": [[[0, 0], [0, 1], [1, 1]], [[0, 0], [1, 0], [1, 1]]],
            "scores": [[10, 11, 12], [20, 21, 22]],
            "timestamps": [[100, 101, 102], [200, 201, 202]],
            "done": [[False, True, False], [False, False, True]],
        }

        transitions = list(normalize_decoded_batch(batch))

        self.assertEqual(len(transitions), 6)
        self.assertEqual(
            [(transition.gameid, transition.step) for transition in transitions],
            [(7, 0), (7, 1), (8, 2), (9, 0), (9, 1), (9, 2)],
        )
        self.assertEqual(
            [transition.sequence_id for transition in transitions],
            ["7:100", "7:100", "7:100", "9:200", "9:200", "9:200"],
        )
        self.assertEqual(
            [transition.sequence_step for transition in transitions],
            [0, 1, 2, 0, 1, 2],
        )
        self.assertEqual(
            transitions[0].next_observation["tty_chars"], [[46, 64], [46, 46]]
        )
        self.assertIsNone(transitions[1].next_observation)
        self.assertIsNone(transitions[2].next_observation)
        self.assertEqual(transitions[3].next_observation["tty_cursor"], [1, 0])
        self.assertEqual(transitions[4].next_observation["scores"], 22)
        self.assertIsNone(transitions[5].next_observation)

    def test_render_observation_text_is_stable(self) -> None:
        text = render_observation_text(
            {
                "tty_chars": [[64, 46], [46, 46]],
                "message": [72, 105],
                "blstats": [1, 2, 3],
                "inventory": [],
            }
        )

        self.assertEqual(
            text,
            "MAP:\n@.\n..\nMESSAGE:\nHi\nBLSTATS:\n[1, 2, 3]\nINVENTORY:\n<empty>",
        )

    def test_render_observation_text_can_compact_empty_terminal_rows(self) -> None:
        text = render_observation_text(
            {
                "tty_chars": [
                    [32, 32],
                    [64, 46],
                    [32, 32],
                    [46, 46],
                    [32, 32],
                ],
                "message": [72, 105],
                "blstats": [1, 2, 3],
                "inventory": [],
            },
            compact_map=True,
        )

        self.assertEqual(
            text,
            "MAP:\n@.\n..\nMESSAGE:\nHi\nBLSTATS:\n[1, 2, 3]\nINVENTORY:\n<empty>",
        )

    def test_build_policy_and_next_frame_rows(self) -> None:
        manifest = _manifest()
        transition = _decoded_transitions()[0]
        metadata = {"role": "Sam", "race": "Hum", "align": "Law", "death": "quit"}

        policy_row = build_policy_action_row(
            dataset_name="fixture-nld",
            split="train",
            mode="single_frame",
            transition=transition,
            action_manifest=manifest,
            game_metadata=metadata,
            history=[],
        )
        next_frame_row = build_next_frame_row(
            dataset_name="fixture-nld",
            split="train",
            mode="single_frame",
            transition=transition,
            action_manifest=manifest,
            game_metadata=metadata,
            history=[],
        )

        self.assertEqual(policy_row["task"], "policy_action")
        self.assertEqual(
            json.loads(policy_row["messages"][2]["content"]), {"action_id": 0}
        )
        self.assertEqual(next_frame_row["task"], "next_frame")
        self.assertEqual(
            next_frame_row["metadata"]["target_frame_kind"],
            "compact_rendered_observation_text",
        )
        self.assertEqual(
            next_frame_row["metadata"]["next_frame_response_format"],
            "raw_frame",
        )
        self.assertTrue(next_frame_row["messages"][2]["content"].startswith("MAP:\n"))

    def test_build_next_frame_row_persists_nld_sequence_fields(self) -> None:
        manifest = _manifest()
        transition = list(
            normalize_decoded_batch(
                {
                    "gameids": [[1, 1]],
                    "keypresses": [[107, 106]],
                    "timestamps": [[1234, 1235]],
                    "tty_chars": [
                        [
                            [[64, 46], [46, 46]],
                            [[46, 64], [46, 46]],
                        ]
                    ],
                }
            )
        )[0]

        row = build_next_frame_row(
            dataset_name="fixture-nld",
            split="train",
            mode="single_frame",
            transition=transition,
            action_manifest=manifest,
            game_metadata={},
            history=[],
        )

        self.assertEqual(row["sequence_id"], "fixture-nld:1:1234")
        self.assertEqual(row["sequence_step"], 0)

    def test_build_pseudo_policy_row_marks_label_provenance(self) -> None:
        transition = list(
            normalize_frame_only_batch(
                {
                    "gameids": [1, 1],
                    "tty_chars": [
                        [[64, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                    ],
                }
            )
        )[0]
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=1,
                    nle_action_name="CompassDirection.E",
                    raw_key_code=108,
                    key_label="l",
                ),
            ),
        )
        label = infer_visible_movement_pseudo_label(
            transition=transition,
            action_manifest=manifest,
        )
        assert label is not None

        row = build_pseudo_policy_action_row(
            dataset_name="fixture-nld",
            split="train",
            mode="single_frame",
            transition=transition,
            action_manifest=manifest,
            game_metadata={"role": "Sam"},
            history=[],
            pseudo_label=label,
        )

        self.assertEqual(row["task"], "policy_action")
        self.assertEqual(json.loads(row["messages"][2]["content"]), {"action_id": 1})
        self.assertEqual(row["metadata"]["label_source"], "pseudo_visible_player_delta")
        self.assertEqual(row["metadata"]["label_confidence"], 1.0)
        self.assertFalse(row["metadata"]["true_keypress_label_available"])

    def test_build_pseudo_next_frame_row_marks_conditioning_action_provenance(
        self,
    ) -> None:
        transition = list(
            normalize_frame_only_batch(
                {
                    "gameids": [1, 1],
                    "tty_chars": [
                        [[64, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                    ],
                }
            )
        )[0]
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=1,
                    nle_action_name="CompassDirection.E",
                    raw_key_code=108,
                    key_label="l",
                ),
            ),
        )
        label = infer_visible_movement_pseudo_label(
            transition=transition,
            action_manifest=manifest,
        )
        assert label is not None

        row = build_pseudo_next_frame_row(
            dataset_name="fixture-nld",
            split="train",
            mode="single_frame",
            transition=transition,
            action_manifest=manifest,
            game_metadata={"role": "Sam"},
            history=[],
            pseudo_label=label,
        )

        self.assertEqual(row["task"], "next_frame")
        self.assertEqual(row["metadata"]["conditioning_action_id"], 1)
        self.assertEqual(row["metadata"]["label_source"], "pseudo_visible_player_delta")
        self.assertEqual(
            row["metadata"]["target_frame_kind"],
            "compact_rendered_observation_text",
        )
        self.assertEqual(
            row["metadata"]["next_frame_response_format"],
            "raw_frame",
        )
        self.assertTrue(row["messages"][2]["content"].startswith("MAP:\n"))

    def test_history_buffer_never_crosses_episode(self) -> None:
        buffer = HistoryBuffer(max_items=4)
        buffer.append(gameid=1, observation_text="old", action_id=0)

        self.assertEqual(
            buffer.history_for(gameid=2, mode="context_4", token_budget=1000), []
        )

    def test_feedback_history_buffer_never_crosses_episode(self) -> None:
        buffer = HistoryBuffer(max_items=4)
        buffer.append(
            gameid=1,
            observation_text="old",
            action_id=0,
            feedback={"action_id": 0, "message": "Moved"},
        )

        self.assertEqual(
            buffer.history_for(
                gameid=2,
                mode="feedback_context_4",
                token_budget=1000,
            ),
            [],
        )

    def test_feedback_context_rows_use_prior_outcome_without_future_leakage(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            write_sft_dataset(
                dataset_name="fixture-nld",
                mode="feedback_context_1",
                transitions=_decoded_transitions(),
                action_manifest=_manifest(),
                game_metadata_by_id={1: {"role": "Sam"}},
                splits={"train": {1, 2}, "validation": set(), "test": set()},
                out_dir=out_dir,
                max_rows=10,
                tasks=("policy_action",),
            )

            policy_rows = [
                json.loads(line)
                for line in (out_dir / "train.policy_action.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        first_prompt = policy_rows[0]["messages"][1]["content"]
        second_prompt = policy_rows[1]["messages"][1]["content"]
        self.assertNotIn("Recent action feedback:", first_prompt)
        self.assertIn("Recent action feedback:", second_prompt)
        self.assertIn("action_id=0", second_prompt)
        self.assertIn('message="Moved"', second_prompt)
        self.assertNotIn("Block", second_prompt)

    def test_feedback_context_next_frame_prompt_does_not_leak_target_frame(
        self,
    ) -> None:
        transitions = list(
            normalize_decoded_batch(
                {
                    "gameids": [1, 1, 1],
                    "steps": [0, 1, 2],
                    "keypresses": [107, 106, 107],
                    "tty_chars": [
                        [[64, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                        [[46, 46], [64, 46]],
                    ],
                    "message": [
                        [72, 105],
                        [77, 111, 118, 101, 100],
                        [68, 111, 110, 101],
                    ],
                    "blstats": [[0] * 13, [0] * 13, [0] * 13],
                }
            )
        )
        manifest = _manifest()
        buffer = HistoryBuffer(max_items=4)
        action_id = manifest.action_id_for_raw_key(transitions[0].raw_key_code)
        buffer.append(
            gameid=transitions[0].gameid,
            observation_text=render_observation_text(transitions[0].observation),
            action_id=action_id,
            feedback=policy_feedback_from_outcome_observation(
                action_id=action_id,
                observation=transitions[0].next_observation,
            ),
        )

        row = build_next_frame_row(
            dataset_name="fixture-nld",
            split="train",
            mode="feedback_context_1",
            transition=transitions[1],
            action_manifest=manifest,
            game_metadata={},
            history=buffer.history_for(
                gameid=1,
                mode="feedback_context_1",
                token_budget=1000,
            ),
        )

        prompt = row["messages"][1]["content"]
        self.assertIn("Recent action feedback:", prompt)
        self.assertIn('message="Moved"', prompt)
        self.assertNotIn("Done", prompt)
        self.assertIn("Done", row["messages"][2]["content"])

    def test_write_sft_dataset_outputs_multitask_jsonl_and_reports(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = write_sft_dataset(
                dataset_name="fixture-nld",
                mode="single_frame",
                transitions=_decoded_transitions(),
                action_manifest=_manifest(),
                game_metadata_by_id={
                    1: {"role": "Sam", "race": "Hum", "align": "Law", "death": "quit"}
                },
                splits={"train": {1, 2}, "validation": set(), "test": set()},
                out_dir=out_dir,
                max_rows=10,
                tasks=("policy_action", "next_frame"),
            )

            train_rows = [
                json.loads(line)
                for line in (out_dir / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(result.accepted_policy_rows, 2)
            self.assertEqual(result.accepted_next_frame_rows, 1)
            self.assertEqual(result.rejected_rows, 2)
            self.assertTrue((out_dir / "train.policy_action.jsonl").exists())
            self.assertTrue((out_dir / "train.next_frame.jsonl").exists())
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertTrue((out_dir / "rejection_report.json").exists())
            self.assertEqual(
                {row["task"] for row in train_rows}, {"policy_action", "next_frame"}
            )

    def test_merge_sft_dataset_shards_writes_trainable_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_one = root / "shard-one"
            shard_two = root / "shard-two"
            out_dir = root / "merged"
            for shard_dir, action_id in ((shard_one, 1), (shard_two, 2)):
                shard_dir.mkdir()
                row = {
                    "split": "train",
                    "task": "policy_action",
                    "messages": [],
                    "metadata": {"target_action_id": action_id},
                }
                (shard_dir / "train.jsonl").write_text(
                    json.dumps(row, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (shard_dir / "train.policy_action.jsonl").write_text(
                    json.dumps(row, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                for name in (
                    "train.next_frame.jsonl",
                    "validation.jsonl",
                    "validation.policy_action.jsonl",
                    "validation.next_frame.jsonl",
                    "test.jsonl",
                    "test.policy_action.jsonl",
                    "test.next_frame.jsonl",
                    "sample_rows.jsonl",
                ):
                    (shard_dir / name).write_text("", encoding="utf-8")
                (shard_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "learn-nethack.sft-manifest.v1",
                            "dataset_name": "fixture-nld",
                            "mode": "single_frame",
                            "tasks": ["policy_action", "next_frame"],
                            "label_source": "pseudo_visible_player_delta",
                            "env_id": "NetHack-v0",
                            "accepted_policy_rows": 1,
                            "accepted_next_frame_rows": 0,
                            "rejected_rows": action_id,
                            "next_frame_status": "no_rows",
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                (shard_dir / "rejection_report.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "learn-nethack.sft-rejections.v1",
                            "total_rejected": action_id,
                            "reasons": {"pseudo_label_unavailable": action_id},
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                (shard_dir / "split_manifest.json").write_text(
                    json.dumps(
                        {"train": [action_id], "validation": [], "test": []},
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                _manifest().save(shard_dir / "action_manifest.json")

            result = merge_sft_dataset_shards(
                shard_dirs=[shard_one, shard_two],
                out_dir=out_dir,
            )

            merged_rows = [
                json.loads(line)
                for line in (out_dir / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
            rejection_report = json.loads(
                (out_dir / "rejection_report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.accepted_policy_rows, 2)
        self.assertEqual(result.accepted_next_frame_rows, 0)
        self.assertEqual(result.rejected_rows, 3)
        self.assertEqual(
            [row["metadata"]["target_action_id"] for row in merged_rows], [1, 2]
        )
        self.assertEqual(manifest["accepted_policy_rows"], 2)
        self.assertEqual(manifest["shard_count"], 2)
        self.assertEqual(rejection_report["total_rejected"], 3)
        self.assertEqual(
            rejection_report["reasons"],
            {"pseudo_label_unavailable": 3},
        )

    def test_write_sft_dataset_can_output_next_frame_rows_without_policy_task(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = write_sft_dataset(
                dataset_name="fixture-nld",
                mode="single_frame",
                transitions=_decoded_transitions(),
                action_manifest=_manifest(),
                game_metadata_by_id={1: {"role": "Sam"}},
                splits={"train": {1, 2}, "validation": set(), "test": set()},
                out_dir=out_dir,
                max_rows=10,
                tasks=("next_frame",),
            )

            next_frame_rows = [
                json.loads(line)
                for line in (out_dir / "train.next_frame.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

            self.assertEqual(result.accepted_policy_rows, 0)
            self.assertEqual(result.accepted_next_frame_rows, 1)
            self.assertEqual(next_frame_rows[0]["task"], "next_frame")
            self.assertEqual(
                next_frame_rows[0]["metadata"]["conditioning_action_id"], 0
            )

    def test_write_pseudo_label_policy_dataset_outputs_provenance_report(
        self,
    ) -> None:
        transitions = list(
            normalize_frame_only_batch(
                {
                    "gameids": [1, 1, 1],
                    "tty_chars": [
                        [[64, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                        [[46, 46], [46, 64]],
                    ],
                }
            )
        )
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=1,
                    nle_action_name="CompassDirection.E",
                    raw_key_code=108,
                    key_label="l",
                ),
                ActionEntry(
                    action_id=2,
                    nle_action_name="CompassDirection.S",
                    raw_key_code=106,
                    key_label="j",
                ),
            ),
        )
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = write_pseudo_label_policy_dataset(
                dataset_name="fixture-nld",
                mode="single_frame",
                transitions=transitions,
                action_manifest=manifest,
                game_metadata_by_id={1: {"role": "Sam"}},
                splits={"train": {1}, "validation": set(), "test": set()},
                out_dir=out_dir,
                max_rows=10,
            )
            manifest_payload = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
            rows = [
                json.loads(line)
                for line in (out_dir / "train.policy_action.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(result.accepted_policy_rows, 2)
        self.assertEqual(result.accepted_next_frame_rows, 0)
        self.assertEqual(result.rejected_rows, 1)
        self.assertEqual(
            manifest_payload["label_source"], "pseudo_visible_player_delta"
        )
        self.assertEqual(
            rows[0]["metadata"]["label_source"], "pseudo_visible_player_delta"
        )

    def test_write_pseudo_label_policy_dataset_can_output_next_frame_rows(
        self,
    ) -> None:
        transitions = list(
            normalize_frame_only_batch(
                {
                    "gameids": [1, 1, 1],
                    "tty_chars": [
                        [[64, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                        [[46, 46], [46, 64]],
                    ],
                }
            )
        )
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=1,
                    nle_action_name="CompassDirection.E",
                    raw_key_code=108,
                    key_label="l",
                ),
                ActionEntry(
                    action_id=2,
                    nle_action_name="CompassDirection.S",
                    raw_key_code=106,
                    key_label="j",
                ),
            ),
        )
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = write_pseudo_label_policy_dataset(
                dataset_name="fixture-nld",
                mode="single_frame",
                transitions=transitions,
                action_manifest=manifest,
                game_metadata_by_id={1: {"role": "Sam"}},
                splits={"train": {1}, "validation": set(), "test": set()},
                out_dir=out_dir,
                max_rows=10,
                tasks=("policy_action", "next_frame"),
            )
            manifest_payload = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )
            next_frame_rows = [
                json.loads(line)
                for line in (out_dir / "train.next_frame.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(result.accepted_policy_rows, 2)
        self.assertEqual(result.accepted_next_frame_rows, 2)
        self.assertEqual(manifest_payload["tasks"], ["policy_action", "next_frame"])
        self.assertEqual(
            next_frame_rows[0]["metadata"]["label_source"],
            "pseudo_visible_player_delta",
        )

    def test_write_pseudo_label_policy_dataset_reports_progress(
        self,
    ) -> None:
        transitions = list(
            normalize_frame_only_batch(
                {
                    "gameids": [1, 1, 1],
                    "tty_chars": [
                        [[64, 46], [46, 46]],
                        [[46, 64], [46, 46]],
                        [[46, 46], [46, 64]],
                    ],
                }
            )
        )
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=1,
                    nle_action_name="CompassDirection.E",
                    raw_key_code=108,
                    key_label="l",
                ),
                ActionEntry(
                    action_id=2,
                    nle_action_name="CompassDirection.S",
                    raw_key_code=106,
                    key_label="j",
                ),
            ),
        )
        progress = []
        with TemporaryDirectory() as tmp:
            write_pseudo_label_policy_dataset(
                dataset_name="fixture-nld",
                mode="single_frame",
                transitions=transitions,
                action_manifest=manifest,
                game_metadata_by_id={1: {"role": "Sam"}},
                splits={"train": {1}, "validation": set(), "test": set()},
                out_dir=Path(tmp),
                max_rows=10,
                tasks=("policy_action",),
                progress_callback=progress.append,
                progress_interval=2,
            )

        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].accepted_policy_rows, 2)
        self.assertEqual(progress[0].rejected_rows, 0)
        self.assertEqual(progress[0].reason, "accepted")

    def test_build_pseudo_label_sft_from_frame_batches_splits_by_episode(
        self,
    ) -> None:
        batch = {
            "gameids": [1, 1, 2, 2],
            "tty_chars": [
                [[64, 46], [46, 46]],
                [[46, 64], [46, 46]],
                [[64, 46], [46, 46]],
                [[46, 46], [64, 46]],
            ],
        }
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=1,
                    nle_action_name="CompassDirection.E",
                    raw_key_code=108,
                    key_label="l",
                ),
                ActionEntry(
                    action_id=2,
                    nle_action_name="CompassDirection.S",
                    raw_key_code=106,
                    key_label="j",
                ),
            ),
        )
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = build_pseudo_label_sft_from_frame_batches(
                dataset_name="fixture-nld",
                mode="single_frame",
                batches=[batch],
                action_manifest=manifest,
                gameids=[1, 2],
                game_metadata_by_id={1: {"role": "Sam"}, 2: {"role": "Wiz"}},
                out_dir=out_dir,
                max_rows=10,
                seed=20260615,
            )
            split_manifest = json.loads(
                (out_dir / "split_manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.accepted_policy_rows, 2)
        self.assertEqual(
            sorted(
                split_manifest["train"]
                + split_manifest["test"]
                + split_manifest["validation"]
            ),
            [1, 2],
        )

    def test_write_sft_dataset_streams_rows_without_bulk_jsonl_writer(self) -> None:
        original_write_jsonl = sft_build._write_jsonl

        def fail_bulk_writer(*_args, **_kwargs):
            raise AssertionError("full dataset builds must stream rows")

        sft_build._write_jsonl = fail_bulk_writer
        try:
            with TemporaryDirectory() as tmp:
                out_dir = Path(tmp)
                result = sft_build.write_sft_dataset(
                    dataset_name="fixture-nld",
                    mode="single_frame",
                    transitions=_decoded_transitions(),
                    action_manifest=_manifest(),
                    game_metadata_by_id={1: {}},
                    splits={"train": {1}, "validation": set(), "test": set()},
                    out_dir=out_dir,
                    max_rows=None,
                    tasks=("policy_action", "next_frame"),
                )
        finally:
            sft_build._write_jsonl = original_write_jsonl

        self.assertEqual(result.accepted_policy_rows, 2)

    def test_write_sft_dataset_counts_ambiguous_raw_key_rejections(self) -> None:
        manifest = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=0,
                    nle_action_name="PLUS_A",
                    raw_key_code=43,
                    key_label="+",
                ),
                ActionEntry(
                    action_id=1,
                    nle_action_name="PLUS_B",
                    raw_key_code=43,
                    key_label="+",
                ),
                ActionEntry(
                    action_id=2,
                    nle_action_name="NORTH",
                    raw_key_code=107,
                    key_label="k",
                ),
            ),
        )
        batch = {
            "gameids": [1, 1],
            "steps": [0, 1],
            "keypresses": [43, 107],
            "tty_chars": [
                [[64, 46], [46, 46]],
                [[46, 64], [46, 46]],
            ],
        }
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = write_sft_dataset(
                dataset_name="fixture-nld",
                mode="single_frame",
                transitions=list(normalize_decoded_batch(batch)),
                action_manifest=manifest,
                game_metadata_by_id={1: {}},
                splits={"train": {1}, "validation": set(), "test": set()},
                out_dir=out_dir,
                max_rows=10,
                tasks=("policy_action", "next_frame"),
            )
            rejection_report = json.loads(
                (out_dir / "rejection_report.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.accepted_policy_rows, 1)
        self.assertEqual(
            rejection_report["reasons"],
            {"ambiguous_raw_key_code": 1, "missing_next_observation": 1},
        )

    def test_build_sft_from_decoded_batches_runs_full_fixture_loop(self) -> None:
        batch = {
            "gameids": [1, 1],
            "steps": [0, 1],
            "keypresses": [107, 106],
            "tty_chars": [
                [[64, 46], [46, 46]],
                [[46, 64], [46, 46]],
            ],
            "message": [[72, 105], [77, 111, 118, 101, 100]],
            "blstats": [[1, 2, 3], [2, 2, 3]],
        }
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = build_sft_from_decoded_batches(
                dataset_name="fixture-nld",
                mode="single_frame",
                batches=[batch],
                action_manifest=_manifest(),
                gameids=[1],
                game_metadata_by_id={
                    1: {"role": "Sam", "race": "Hum", "align": "Law", "death": "quit"}
                },
                out_dir=out_dir,
                max_rows=10,
                seed=20260615,
                tasks=("policy_action", "next_frame"),
                token_budget=512,
            )

            manifest = json.loads(
                (out_dir / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.accepted_policy_rows, 2)
        self.assertEqual(manifest["token_budget"], 512)
        self.assertEqual(result.accepted_next_frame_rows, 1)
        self.assertEqual(manifest["dataset_name"], "fixture-nld")

    def test_policy_and_next_frame_metrics(self) -> None:
        policy_metrics = compute_policy_metrics(
            predictions=[{"action_id": 1}, {"action_id": 2}],
            labels=[1, 1],
            valid_action_ids={1, 2, 3},
            metadata=[{"role": "Sam"}, {"role": "Wiz"}],
        )
        frame_metrics = compute_next_frame_metrics(
            predictions=["MAP:\n@.\nMESSAGE:\nHi"],
            labels=["MAP:\n@.\nMESSAGE:\nHi"],
        )

        self.assertEqual(policy_metrics["parse_valid_rate"], 1.0)
        self.assertEqual(policy_metrics["action_space_valid_rate"], 1.0)
        self.assertEqual(policy_metrics["exact_match_rate"], 0.5)
        self.assertEqual(policy_metrics["macro_action_accuracy"], 0.5)
        self.assertEqual(policy_metrics["predicted_dominant_action_rate"], 0.5)
        self.assertEqual(policy_metrics["dominant_action_collapse"], 0.0)
        self.assertEqual(policy_metrics["role_exact_match/Sam"], 1.0)
        self.assertEqual(policy_metrics["role_exact_match/Wiz"], 0.0)
        self.assertEqual(frame_metrics["next_frame_exact_match_rate"], 1.0)
        self.assertEqual(frame_metrics["next_frame_char_accuracy"], 1.0)

    def test_policy_metrics_expose_dominant_action_collapse(self) -> None:
        metrics = compute_policy_metrics(
            predictions=[{"action_id": 1}] * 4,
            labels=[0, 1, 1, 2],
            valid_action_ids={0, 1, 2},
            metadata=[{"role": "Arc"}] * 4,
        )

        self.assertEqual(metrics["exact_match_rate"], 0.5)
        self.assertAlmostEqual(metrics["macro_action_accuracy"], 1.0 / 3.0)
        self.assertEqual(metrics["non_modal_action_exact_match_rate"], 0.0)
        self.assertEqual(metrics["predicted_unique_action_count"], 1.0)
        self.assertEqual(metrics["predicted_dominant_action_id"], 1.0)
        self.assertEqual(metrics["predicted_dominant_action_rate"], 1.0)
        self.assertEqual(metrics["dominant_action_collapse"], 1.0)

    def test_split_row_limits_reserve_validation_and_test_rows(self) -> None:
        manifest = ActionManifest(
            env_id="NetHack-v0",
            entries=(
                ActionEntry(
                    action_id=0,
                    nle_action_name="CompassDirection.N",
                    raw_key_code=107,
                    key_label="k",
                ),
            ),
        )
        observation = {
            "tty_chars": [[64, 46]],
            "message": [],
            "blstats": [0] * 27,
            "inventory": [],
        }
        transitions = [
            types.SimpleNamespace(
                gameid=gameid,
                step=step,
                raw_key_code=107,
                observation=observation,
                next_observation=observation,
                sequence_id=None,
                sequence_step=None,
            )
            for gameid in (1, 2, 3)
            for step in range(3)
        ]

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = write_sft_dataset(
                dataset_name="fixture",
                mode="single_frame",
                transitions=transitions,
                action_manifest=manifest,
                game_metadata_by_id={},
                splits={"train": {1}, "validation": {2}, "test": {3}},
                out_dir=out,
                tasks=("policy_action",),
                split_row_limits=SplitRowLimits(train=2, validation=2, test=2),
            )
            counts = {
                split: len(
                    (out / f"{split}.policy_action.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                for split in ("train", "validation", "test")
            }
            written_manifest = json.loads(
                (out / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result.accepted_policy_rows, 6)
        self.assertEqual(counts, {"train": 2, "validation": 2, "test": 2})
        self.assertEqual(
            written_manifest["accepted_rows_by_split"],
            {
                "train": {"policy_action": 2, "next_frame": 0},
                "validation": {"policy_action": 2, "next_frame": 0},
                "test": {"policy_action": 2, "next_frame": 0},
            },
        )
        self.assertEqual(
            written_manifest["split_quota_skips"],
            {"train": 1, "validation": 1},
        )
        self.assertIs(written_manifest["split_limits_satisfied"], True)

    def test_split_row_limits_write_diagnostic_manifest_then_fail_if_incomplete(
        self,
    ) -> None:
        manifest = ActionManifest(
            env_id="NetHack-v0",
            entries=(
                ActionEntry(
                    action_id=0,
                    nle_action_name="CompassDirection.N",
                    raw_key_code=107,
                    key_label="k",
                ),
            ),
        )
        observation = {
            "tty_chars": [[64, 46]],
            "message": [],
            "blstats": [0] * 27,
            "inventory": [],
        }
        transitions = [
            types.SimpleNamespace(
                gameid=1,
                step=0,
                raw_key_code=107,
                observation=observation,
                next_observation=observation,
                sequence_id=None,
                sequence_step=None,
            )
        ]

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "split row limits not satisfied"):
                write_sft_dataset(
                    dataset_name="fixture",
                    mode="single_frame",
                    transitions=transitions,
                    action_manifest=manifest,
                    game_metadata_by_id={},
                    splits={"train": {1}, "validation": {2}, "test": {3}},
                    out_dir=out,
                    tasks=("policy_action",),
                    split_row_limits=SplitRowLimits(train=1, validation=1, test=1),
                )
            written_manifest = json.loads(
                (out / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertIs(written_manifest["split_limits_satisfied"], False)
        self.assertEqual(
            written_manifest["accepted_rows_by_split"]["train"]["policy_action"],
            1,
        )

    def test_split_row_limits_cannot_be_combined_with_global_limit(self) -> None:
        with (
            TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "max_rows or split_row_limits"),
        ):
            write_sft_dataset(
                dataset_name="fixture",
                mode="single_frame",
                transitions=[],
                action_manifest=ActionManifest(env_id="NetHack-v0", entries=()),
                game_metadata_by_id={},
                splits={"train": set(), "validation": set(), "test": set()},
                out_dir=tmp,
                max_rows=1,
                tasks=("policy_action",),
                split_row_limits=SplitRowLimits(train=1, validation=1, test=1),
            )

    def test_changed_state_metrics_do_not_reward_copying_current_frame(self) -> None:
        current_blstats = [0] * 27
        current_blstats[20] = 10
        next_blstats = list(current_blstats)
        next_blstats[0] = 1
        next_blstats[20] = 11
        current = "\n".join(
            [
                "MAP:",
                "@.",
                "MESSAGE:",
                "You wait.",
                "BLSTATS:",
                json.dumps(current_blstats),
                "INVENTORY:",
                "<empty>",
            ]
        )
        following = "\n".join(
            [
                "MAP:",
                ".@",
                "MESSAGE:",
                "You move east.",
                "BLSTATS:",
                json.dumps(next_blstats),
                "INVENTORY:",
                "<empty>",
            ]
        )

        copy_metrics = compute_next_frame_metrics(
            current_frames=[current],
            predictions=[current],
            labels=[following],
        )
        perfect_metrics = compute_next_frame_metrics(
            current_frames=[current],
            predictions=[following],
            labels=[following],
        )

        self.assertEqual(copy_metrics["next_frame_changed_map_cell_f1"], 0.0)
        self.assertEqual(copy_metrics["next_frame_player_coordinate_exact_rate"], 0.0)
        self.assertEqual(perfect_metrics["next_frame_changed_map_cell_precision"], 1.0)
        self.assertEqual(perfect_metrics["next_frame_changed_map_cell_recall"], 1.0)
        self.assertEqual(perfect_metrics["next_frame_changed_map_cell_f1"], 1.0)
        self.assertEqual(
            perfect_metrics["next_frame_player_coordinate_exact_rate"], 1.0
        )
        self.assertEqual(perfect_metrics["next_frame_blstats_field_exact_rate"], 1.0)
        self.assertEqual(perfect_metrics["next_frame_blstats_numeric_mae"], 0.0)
        self.assertEqual(perfect_metrics["next_frame_game_turn_delta_exact_rate"], 1.0)
        self.assertEqual(
            perfect_metrics["next_frame_message_normalized_exact_rate"], 1.0
        )

    def test_evaluate_policy_rows_scores_constrained_action_candidates(self) -> None:
        policy = AlwaysSouthPolicy()
        rows = [
            {
                "task": "policy_action",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "Current observation:\nMAP:\n@."},
                    {"role": "assistant", "content": '{"action_id": 1}'},
                ],
                "metadata": {
                    "target_action_id": 1,
                    "valid_action_ids": [0, 1],
                    "role": "Sam",
                },
            },
            {
                "task": "policy_action",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "Current observation:\nMAP:\n#@"},
                    {"role": "assistant", "content": '{"action_id": 0}'},
                ],
                "metadata": {
                    "target_action_id": 0,
                    "valid_action_ids": [0, 1],
                    "role": "Wiz",
                },
            },
        ]

        metrics = evaluate_policy_rows_with_policy(rows=rows, policy=policy)

        self.assertEqual(metrics["row_count"], 2.0)
        self.assertEqual(metrics["parse_valid_rate"], 1.0)
        self.assertEqual(metrics["action_space_valid_rate"], 1.0)
        self.assertEqual(metrics["exact_match_rate"], 0.5)
        self.assertEqual(policy.last_observation_text, "MAP:\n#@")

    def test_evaluate_next_frame_rows_scores_model_generated_frames(self) -> None:
        predictor = EchoNextFramePredictor()
        rows = [
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 1}\n'
                        "Current observation:\nMAP:\n@.\nMESSAGE:\nHi",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}
                        ),
                    },
                ],
                "metadata": {
                    "conditioning_action_id": 1,
                    "valid_action_ids": [0, 1],
                },
            },
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 0}\n'
                        "Current observation:\nMAP:\n@.\nMESSAGE:\nHi",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}
                        ),
                    },
                ],
                "metadata": {
                    "conditioning_action_id": 0,
                    "valid_action_ids": [0, 1],
                },
            },
        ]
        samples: list[dict] = []

        metrics = evaluate_next_frame_rows_with_predictor(
            rows=rows,
            predictor=predictor,
            sample_callback=samples.append,
        )

        self.assertEqual(metrics["next_frame_eval_row_count"], 2.0)
        self.assertEqual(metrics["next_frame_parse_valid_rate"], 1.0)
        self.assertEqual(metrics["next_frame_exact_match_rate"], 0.5)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["phase"], "next_frame_generate")
        self.assertTrue(samples[0]["parse_valid"])
        self.assertIsNone(samples[0]["parser_error"])
        self.assertIsNone(samples[0]["parse_failure_reason"])
        self.assertEqual(samples[0]["label_chars"], len("MAP:\n.@\nMESSAGE:\nMoved"))
        self.assertIn('"next_frame"', samples[0]["raw_output"])
        self.assertEqual(
            predictor.calls[0]["observation_text"], "MAP:\n@.\nMESSAGE:\nHi"
        )
        self.assertEqual(predictor.calls[0]["action_id"], 1)

    def test_evaluate_next_frame_rows_samples_parse_errors(self) -> None:
        class InvalidNextFramePredictor:
            def generate_next_frame_json(
                self,
                *,
                observation_text: str,
                action_id: int,
                history: list[tuple[str, int]],
            ) -> str:
                return "not json"

        rows = [
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 1}\n'
                        "Current observation:\nMAP:\n@.\nMESSAGE:\nHi",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}
                        ),
                    },
                ],
                "metadata": {
                    "conditioning_action_id": 1,
                    "valid_action_ids": [0, 1],
                },
            }
        ]
        samples: list[dict] = []

        metrics = evaluate_next_frame_rows_with_predictor(
            rows=rows,
            predictor=InvalidNextFramePredictor(),
            sample_callback=samples.append,
        )

        self.assertEqual(metrics["next_frame_parse_valid_rate"], 0.0)
        self.assertEqual(metrics["next_frame_parse_failure_invalid_json_rate"], 1.0)
        self.assertEqual(metrics["next_frame_parse_failure_truncated_json_rate"], 0.0)
        self.assertFalse(samples[0]["parse_valid"])
        self.assertEqual(samples[0]["parser_error"], "invalid next-frame JSON")
        self.assertEqual(samples[0]["parse_failure_reason"], "invalid_json")
        self.assertEqual(samples[0]["raw_output"], "not json")

    def test_evaluate_next_frame_rows_counts_truncated_json_parse_errors(
        self,
    ) -> None:
        class TruncatedNextFramePredictor:
            def generate_next_frame_json(
                self,
                *,
                observation_text: str,
                action_id: int,
                history: list[tuple[str, int]],
            ) -> str:
                return '{"next_frame": "MAP:\\n@'

        rows = [
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 1}\n'
                        "Current observation:\nMAP:\n@.\nMESSAGE:\nHi",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}
                        ),
                    },
                ],
                "metadata": {
                    "conditioning_action_id": 1,
                    "valid_action_ids": [0, 1],
                },
            }
        ]
        samples: list[dict] = []

        metrics = evaluate_next_frame_rows_with_predictor(
            rows=rows,
            predictor=TruncatedNextFramePredictor(),
            sample_callback=samples.append,
        )

        self.assertEqual(metrics["next_frame_parse_valid_rate"], 0.0)
        self.assertEqual(metrics["next_frame_parse_failure_truncated_json_rate"], 1.0)
        self.assertEqual(metrics["next_frame_parse_failure_invalid_json_rate"], 0.0)
        self.assertEqual(samples[0]["parse_failure_reason"], "truncated_json")
        self.assertEqual(
            samples[0]["raw_output_chars"], len('{"next_frame": "MAP:\\n@')
        )

    def test_evaluate_next_frame_rows_accepts_raw_frame_predictions(self) -> None:
        class RawFramePredictor:
            def generate_next_frame_json(
                self,
                *,
                observation_text: str,
                action_id: int,
                history: list[tuple[str, int]],
            ) -> str:
                return "MAP:\n.@\nMESSAGE:\nMoved"

        rows = [
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 1}\n'
                        "Current observation:\nMAP:\n@.\nMESSAGE:\nHi",
                    },
                    {"role": "assistant", "content": "MAP:\n.@\nMESSAGE:\nMoved"},
                ],
                "metadata": {
                    "conditioning_action_id": 1,
                    "next_frame_response_format": "raw_frame",
                    "valid_action_ids": [0, 1],
                },
            }
        ]
        samples: list[dict] = []

        metrics = evaluate_next_frame_rows_with_predictor(
            rows=rows,
            predictor=RawFramePredictor(),
            sample_callback=samples.append,
        )

        self.assertEqual(metrics["next_frame_parse_valid_rate"], 1.0)
        self.assertEqual(metrics["next_frame_exact_match_rate"], 1.0)
        self.assertEqual(metrics["next_frame_char_accuracy"], 1.0)
        self.assertTrue(samples[0]["parse_valid"])
        self.assertEqual(samples[0]["prediction"], "MAP:\n.@\nMESSAGE:\nMoved")

    def test_evaluate_next_frame_rows_scores_teacher_forced_targets(self) -> None:
        scorer = FixedNextFrameScorer()
        rows = [
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 1}\n'
                        "Current observation:\nMAP:\n@.\nMESSAGE:\nHi",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}
                        ),
                    },
                ],
                "metadata": {"conditioning_action_id": 1},
            },
            {
                "task": "next_frame",
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 0}\n'
                        "Current observation:\nMAP:\n@.\nMESSAGE:\nHi",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}
                        ),
                    },
                ],
                "metadata": {"conditioning_action_id": 0},
            },
        ]

        metrics = evaluate_next_frame_rows_with_scorer(rows=rows, scorer=scorer)

        self.assertEqual(metrics["next_frame_teacher_forced_row_count"], 2.0)
        self.assertEqual(metrics["next_frame_teacher_forced_token_count"], 6.0)
        self.assertEqual(metrics["next_frame_teacher_forced_mean_nll"], 1.0)
        self.assertAlmostEqual(
            metrics["next_frame_teacher_forced_token_accuracy"], 4.0 / 6.0
        )
        self.assertEqual(
            scorer.calls[0]["target_response"],
            json.dumps({"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}),
        )

    def test_evaluate_next_frame_sequences_rolls_predictions_forward(self) -> None:
        predictor = EchoNextFramePredictor()
        rows = [
            {
                "task": "next_frame",
                "episode_id": "fixture:1",
                "step": step,
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": 'Action taken: {"action_id": 1}\n'
                        f"Current observation:\nMAP:\n@{step}\nMESSAGE:\nHi",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"next_frame": "MAP:\n.@\nMESSAGE:\nMoved"}
                        ),
                    },
                ],
                "metadata": {"conditioning_action_id": 1},
            }
            for step in range(3)
        ]
        samples: list[dict] = []
        progress_events: list[dict] = []

        metrics = evaluate_next_frame_sequences_with_predictor(
            rows=rows,
            predictor=predictor,
            horizons=(1, 2),
            progress_callback=progress_events.append,
            sample_callback=samples.append,
        )

        self.assertEqual(metrics["next_1_frame_sequence_window_count"], 3.0)
        self.assertEqual(
            metrics["next_2_frame_sequence_available_window_count"],
            2.0,
        )
        self.assertEqual(
            metrics["next_2_frame_sequence_available_frame_count"],
            4.0,
        )
        self.assertEqual(
            metrics["next_2_frame_sequence_eligible_segment_count"],
            1.0,
        )
        self.assertEqual(metrics["next_2_frame_sequence_window_count"], 2.0)
        self.assertEqual(metrics["next_2_frame_sequence_frame_count"], 4.0)
        self.assertEqual(metrics["next_2_frame_sequence_parse_valid_rate"], 1.0)
        self.assertEqual(metrics["next_2_frame_sequence_exact_match_rate"], 1.0)
        self.assertTrue(samples)
        self.assertEqual(samples[0]["phase"], "next_frame_sequence")
        self.assertEqual(samples[0]["horizon"], 1)
        self.assertTrue(samples[0]["parse_valid"])
        self.assertEqual(
            predictor.calls[4]["observation_text"],
            "MAP:\n.@\nMESSAGE:\nMoved",
        )
        self.assertEqual(
            predictor.calls[4]["history"],
            [("MAP:\n@0\nMESSAGE:\nHi", 1)],
        )
        self.assertIn(
            "next_frame_sequence_frame",
            {event["phase"] for event in progress_events},
        )

    def test_summarize_next_frame_sequence_rows_respects_episode_step_gaps(
        self,
    ) -> None:
        rows = [
            {
                "task": "next_frame",
                "episode_id": episode,
                "step": step,
                "messages": [
                    {"role": "system", "content": "system"},
                    {
                        "role": "user",
                        "content": f'Action taken: {{"action_id": 1}}\n'
                        f"Current observation:\nMAP:\n@{step}",
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({"next_frame": f"MAP:\n.{step}"}),
                    },
                ],
                "metadata": {"conditioning_action_id": 1},
            }
            for episode, step in (
                ("fixture:1", 0),
                ("fixture:1", 1),
                ("fixture:1", 4),
                ("fixture:2", 0),
                ("fixture:2", 1),
                ("fixture:2", 2),
            )
        ]

        metrics = summarize_next_frame_sequence_rows(
            rows=rows,
            horizons=(1, 2, 3, 10),
        )

        self.assertEqual(metrics["next_frame_sequence_row_count"], 6.0)
        self.assertEqual(metrics["next_frame_sequence_episode_count"], 2.0)
        self.assertEqual(metrics["next_frame_sequence_segment_count"], 3.0)
        self.assertEqual(metrics["next_frame_sequence_max_segment_length"], 3.0)
        self.assertEqual(metrics["next_1_frame_sequence_available_window_count"], 6.0)
        self.assertEqual(metrics["next_2_frame_sequence_available_window_count"], 3.0)
        self.assertEqual(metrics["next_3_frame_sequence_available_window_count"], 1.0)
        self.assertEqual(metrics["next_10_frame_sequence_available_window_count"], 0.0)

    def test_score_to_beat_report_declares_improvement_verdict(self) -> None:
        report = build_score_to_beat_report(
            baseline_metrics={
                "exact_match_rate": 0.25,
                "next_frame_char_accuracy": 0.50,
            },
            trained_metrics={
                "exact_match_rate": 0.40,
                "next_frame_char_accuracy": 0.49,
            },
            baseline_run_id="base-gemma",
            trained_run_id="sft-full",
        )

        self.assertEqual(report["verdict"], "mixed")
        self.assertTrue(report["metrics"]["exact_match_rate"]["improved"])
        self.assertFalse(report["metrics"]["next_frame_char_accuracy"]["improved"])
        self.assertEqual(report["metrics"]["exact_match_rate"]["delta"], 0.15)

    def test_score_to_beat_report_treats_next_frame_nll_as_lower_is_better(
        self,
    ) -> None:
        report = build_score_to_beat_report(
            baseline_metrics={"next_frame_teacher_forced_mean_nll": 14.0},
            trained_metrics={"next_frame_teacher_forced_mean_nll": 13.5},
            baseline_run_id="base-gemma",
            trained_run_id="sft-full",
        )

        self.assertEqual(report["verdict"], "improved")
        self.assertTrue(
            report["metrics"]["next_frame_teacher_forced_mean_nll"]["improved"]
        )
        self.assertEqual(
            report["metrics"]["next_frame_teacher_forced_mean_nll"]["direction"],
            "lower_is_better",
        )

    def test_score_to_beat_report_compares_generated_next_frame_parse_validity(
        self,
    ) -> None:
        report = build_score_to_beat_report(
            baseline_metrics={
                "next_frame_parse_valid_rate": 1.0,
                "next_frame_teacher_forced_mean_nll": 1.0,
                "next_frame_map_line_exact_rate": 0.25,
                "next_frame_message_exact_rate": 0.5,
            },
            trained_metrics={
                "next_frame_parse_valid_rate": 0.0,
                "next_frame_map_line_exact_rate": 0.25,
                "next_frame_message_exact_rate": 0.75,
            },
            baseline_run_id="base-gemma",
            trained_run_id="sft-full",
        )

        self.assertEqual(report["verdict"], "mixed")
        self.assertTrue(report["metrics"]["next_frame_parse_valid_rate"]["regressed"])
        self.assertFalse(
            report["metrics"]["next_frame_map_line_exact_rate"]["regressed"]
        )
        self.assertTrue(report["metrics"]["next_frame_message_exact_rate"]["improved"])

    def test_score_to_beat_report_compares_next_n_frame_sequence_metrics(
        self,
    ) -> None:
        report = build_score_to_beat_report(
            baseline_metrics={
                "next_5_frame_sequence_char_accuracy": 0.25,
                "next_5_frame_sequence_window_count": 2.0,
                "next_5_frame_sequence_frame_count": 10.0,
            },
            trained_metrics={
                "next_5_frame_sequence_char_accuracy": 0.40,
                "next_5_frame_sequence_window_count": 2.0,
                "next_5_frame_sequence_frame_count": 10.0,
            },
            baseline_run_id="base-gemma",
            trained_run_id="sft-full",
        )

        self.assertEqual(report["verdict"], "improved")
        self.assertTrue(
            report["metrics"]["next_5_frame_sequence_char_accuracy"]["improved"]
        )

    def test_score_to_beat_report_marks_missing_next_n_windows_unproven(self) -> None:
        report = build_score_to_beat_report(
            baseline_metrics={
                "next_10_frame_sequence_available_window_count": 0.0,
                "next_10_frame_sequence_available_frame_count": 0.0,
                "next_10_frame_sequence_char_accuracy": 0.0,
                "next_10_frame_sequence_window_count": 0.0,
                "next_10_frame_sequence_frame_count": 0.0,
            },
            trained_metrics={
                "next_10_frame_sequence_available_window_count": 0.0,
                "next_10_frame_sequence_available_frame_count": 0.0,
                "next_10_frame_sequence_char_accuracy": 0.5,
                "next_10_frame_sequence_window_count": 0.0,
                "next_10_frame_sequence_frame_count": 0.0,
            },
            baseline_run_id="base-gemma",
            trained_run_id="sft-full",
        )

        self.assertEqual(report["verdict"], "unproven")
        self.assertEqual(
            report["proof_failures"][0]["reason"],
            "zero_available_sequence_evidence",
        )
        self.assertEqual(report["proof_failures"][0]["horizon"], 10)

    def test_score_to_beat_report_records_missing_baseline(self) -> None:
        report = build_score_to_beat_report(
            baseline_metrics=None,
            trained_metrics={"exact_match_rate": 0.40},
            baseline_run_id=None,
            trained_run_id="sft-full",
        )

        self.assertEqual(
            report["score_to_beat_status"],
            "base_gemma_baseline_not_recorded",
        )
        self.assertEqual(report["verdict"], "unproven")

    def test_training_proof_gate_passes_only_when_offline_and_watch_improve(
        self,
    ) -> None:
        score_report = build_score_to_beat_report(
            baseline_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.30,
                "next_frame_parse_valid_rate": 1.0,
                "next_frame_teacher_forced_mean_nll": 1.0,
                "next_1_frame_sequence_available_window_count": 2.0,
                "next_1_frame_sequence_available_frame_count": 2.0,
                "next_1_frame_sequence_window_count": 2.0,
                "next_1_frame_sequence_frame_count": 2.0,
                "next_1_frame_sequence_parse_valid_rate": 1.0,
                "next_1_frame_sequence_char_accuracy": 0.50,
                "next_1_frame_sequence_exact_match_rate": 0.0,
                "next_1_frame_sequence_changed_map_cell_f1": 0.20,
                "next_1_frame_sequence_player_coordinate_exact_rate": 0.50,
                "next_1_frame_sequence_blstats_field_exact_rate": 0.50,
                "next_1_frame_sequence_message_edit_similarity": 0.50,
                "next_5_frame_sequence_available_window_count": 2.0,
                "next_5_frame_sequence_available_frame_count": 10.0,
                "next_5_frame_sequence_window_count": 2.0,
                "next_5_frame_sequence_frame_count": 10.0,
                "next_5_frame_sequence_parse_valid_rate": 1.0,
                "next_5_frame_sequence_char_accuracy": 0.40,
                "next_5_frame_sequence_exact_match_rate": 0.0,
                "next_5_frame_sequence_changed_map_cell_f1": 0.15,
                "next_5_frame_sequence_player_coordinate_exact_rate": 0.50,
                "next_5_frame_sequence_blstats_field_exact_rate": 0.50,
                "next_5_frame_sequence_message_edit_similarity": 0.50,
                "next_10_frame_sequence_available_window_count": 2.0,
                "next_10_frame_sequence_available_frame_count": 20.0,
                "next_10_frame_sequence_window_count": 2.0,
                "next_10_frame_sequence_frame_count": 20.0,
                "next_10_frame_sequence_parse_valid_rate": 1.0,
                "next_10_frame_sequence_char_accuracy": 0.30,
                "next_10_frame_sequence_exact_match_rate": 0.0,
                "next_10_frame_sequence_changed_map_cell_f1": 0.10,
                "next_10_frame_sequence_player_coordinate_exact_rate": 0.50,
                "next_10_frame_sequence_blstats_field_exact_rate": 0.50,
                "next_10_frame_sequence_message_edit_similarity": 0.50,
            },
            trained_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.31,
                "next_frame_parse_valid_rate": 1.0,
                "next_frame_teacher_forced_mean_nll": 0.9,
                "next_1_frame_sequence_available_window_count": 2.0,
                "next_1_frame_sequence_available_frame_count": 2.0,
                "next_1_frame_sequence_window_count": 2.0,
                "next_1_frame_sequence_frame_count": 2.0,
                "next_1_frame_sequence_parse_valid_rate": 1.0,
                "next_1_frame_sequence_char_accuracy": 0.60,
                "next_1_frame_sequence_exact_match_rate": 0.0,
                "next_1_frame_sequence_changed_map_cell_f1": 0.30,
                "next_1_frame_sequence_player_coordinate_exact_rate": 0.50,
                "next_1_frame_sequence_blstats_field_exact_rate": 0.50,
                "next_1_frame_sequence_message_edit_similarity": 0.50,
                "next_5_frame_sequence_available_window_count": 2.0,
                "next_5_frame_sequence_available_frame_count": 10.0,
                "next_5_frame_sequence_window_count": 2.0,
                "next_5_frame_sequence_frame_count": 10.0,
                "next_5_frame_sequence_parse_valid_rate": 1.0,
                "next_5_frame_sequence_char_accuracy": 0.50,
                "next_5_frame_sequence_exact_match_rate": 0.0,
                "next_5_frame_sequence_changed_map_cell_f1": 0.25,
                "next_5_frame_sequence_player_coordinate_exact_rate": 0.50,
                "next_5_frame_sequence_blstats_field_exact_rate": 0.50,
                "next_5_frame_sequence_message_edit_similarity": 0.50,
                "next_10_frame_sequence_available_window_count": 2.0,
                "next_10_frame_sequence_available_frame_count": 20.0,
                "next_10_frame_sequence_window_count": 2.0,
                "next_10_frame_sequence_frame_count": 20.0,
                "next_10_frame_sequence_parse_valid_rate": 1.0,
                "next_10_frame_sequence_char_accuracy": 0.45,
                "next_10_frame_sequence_exact_match_rate": 0.0,
                "next_10_frame_sequence_changed_map_cell_f1": 0.20,
                "next_10_frame_sequence_player_coordinate_exact_rate": 0.50,
                "next_10_frame_sequence_blstats_field_exact_rate": 0.50,
                "next_10_frame_sequence_message_edit_similarity": 0.50,
            },
            baseline_run_id="base",
            trained_run_id="trained",
        )
        watch_report = {
            "run_id": "watch",
            "seed_count": 16,
            "paired_initial_state_equal_count": 16,
            "paired_initial_state_equal": True,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    **_clean_watch_side(
                        fitness_score=1.0,
                        cumulative_reward=0.0,
                        score_delta=0.0,
                    )
                },
                "current": {**_clean_watch_side()},
                "deltas": {
                    "fitness_score": 2.0,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": -0.2,
                    "bad_message_rate": -0.2,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": -0.1,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 0.0,
                    "score_delta": 1.0,
                    "cumulative_reward": 1.0,
                    "depth_delta": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=score_report,
            watch_report=watch_report,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["verdict"], "proved_improved")

    def test_training_proof_gate_rejects_dirty_current_watch_even_if_delta_improves(
        self,
    ) -> None:
        watch_report = {
            "run_id": "watch",
            "seed_count": 16,
            "paired_initial_state_equal_count": 16,
            "paired_initial_state_equal": True,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    **_clean_watch_side(
                        fitness_score=-5.0,
                        cumulative_reward=0.0,
                        score_delta=0.0,
                        wall_message_rate=0.90,
                        bad_message_rate=0.90,
                        action_repeat_rate=1.0,
                        zero_progress_episode=1.0,
                    )
                },
                "current": {
                    **_clean_watch_side(
                        fitness_score=0.5,
                        cumulative_reward=1.0,
                        score_delta=1.0,
                        wall_message_rate=0.40,
                        bad_message_rate=0.40,
                        action_repeat_rate=0.90,
                        zero_progress_episode=1.0,
                    )
                },
                "deltas": {
                    "fitness_score": 5.5,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": -0.50,
                    "bad_message_rate": -0.50,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": -0.10,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 0.0,
                    "score_delta": 1.0,
                    "cumulative_reward": 1.0,
                    "depth_delta": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=_passing_policy_and_next_frame_score_report(),
            watch_report=watch_report,
        )
        failed_names = {
            requirement["name"]
            for requirement in report["requirements"]
            if requirement["status"] == "failed"
        }

        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "failed")
        self.assertIn("watch_current_wall_message_rate_ceiling", failed_names)
        self.assertIn("watch_current_bad_message_rate_ceiling", failed_names)
        self.assertIn("watch_current_action_repeat_rate_ceiling", failed_names)
        self.assertIn("watch_current_zero_progress_episode_ceiling", failed_names)

    def test_training_proof_gate_rejects_dirty_progress_regression(self) -> None:
        watch_report = {
            "run_id": "watch",
            "seed_count": 16,
            "paired_initial_state_equal_count": 16,
            "paired_initial_state_equal": True,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {**_clean_watch_side(dirty_live_progress_event_count=0.0)},
                "current": {**_clean_watch_side(dirty_live_progress_event_count=1.0)},
                "deltas": {
                    "fitness_score": 1.0,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": 0.0,
                    "bad_message_rate": 0.0,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": 0.0,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 1.0,
                    "score_delta": 1.0,
                    "cumulative_reward": 1.0,
                    "depth_delta": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=_passing_policy_and_next_frame_score_report(),
            watch_report=watch_report,
        )
        failed_names = {
            requirement["name"]
            for requirement in report["requirements"]
            if requirement["status"] == "failed"
        }

        self.assertFalse(report["passed"])
        self.assertIn("watch_dirty_live_progress_event_count", failed_names)

    def test_training_proof_gate_rejects_wall_reduction_without_progress(
        self,
    ) -> None:
        score_report = build_score_to_beat_report(
            baseline_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.27,
                "next_frame_parse_valid_rate": 1.0,
                "next_1_frame_sequence_available_window_count": 4.0,
                "next_1_frame_sequence_available_frame_count": 4.0,
                "next_1_frame_sequence_window_count": 4.0,
                "next_1_frame_sequence_frame_count": 4.0,
                "next_1_frame_sequence_parse_valid_rate": 1.0,
                "next_1_frame_sequence_char_accuracy": 0.39,
                "next_1_frame_sequence_exact_match_rate": 0.0,
                "next_5_frame_sequence_available_window_count": 4.0,
                "next_5_frame_sequence_available_frame_count": 20.0,
                "next_5_frame_sequence_window_count": 4.0,
                "next_5_frame_sequence_frame_count": 20.0,
                "next_5_frame_sequence_parse_valid_rate": 1.0,
                "next_5_frame_sequence_char_accuracy": 0.39,
                "next_5_frame_sequence_exact_match_rate": 0.0,
                "next_10_frame_sequence_available_window_count": 4.0,
                "next_10_frame_sequence_available_frame_count": 40.0,
                "next_10_frame_sequence_window_count": 4.0,
                "next_10_frame_sequence_frame_count": 40.0,
                "next_10_frame_sequence_parse_valid_rate": 1.0,
                "next_10_frame_sequence_char_accuracy": 0.62,
                "next_10_frame_sequence_changed_map_cell_f1": 0.62,
                "next_10_frame_sequence_exact_match_rate": 0.0,
            },
            trained_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.19,
                "next_frame_parse_valid_rate": 1.0,
                "next_1_frame_sequence_available_window_count": 4.0,
                "next_1_frame_sequence_available_frame_count": 4.0,
                "next_1_frame_sequence_window_count": 4.0,
                "next_1_frame_sequence_frame_count": 4.0,
                "next_1_frame_sequence_parse_valid_rate": 1.0,
                "next_1_frame_sequence_char_accuracy": 0.75,
                "next_1_frame_sequence_exact_match_rate": 0.0,
                "next_5_frame_sequence_available_window_count": 4.0,
                "next_5_frame_sequence_available_frame_count": 20.0,
                "next_5_frame_sequence_window_count": 4.0,
                "next_5_frame_sequence_frame_count": 20.0,
                "next_5_frame_sequence_parse_valid_rate": 1.0,
                "next_5_frame_sequence_char_accuracy": 0.59,
                "next_5_frame_sequence_exact_match_rate": 0.0,
                "next_10_frame_sequence_available_window_count": 4.0,
                "next_10_frame_sequence_available_frame_count": 40.0,
                "next_10_frame_sequence_window_count": 4.0,
                "next_10_frame_sequence_frame_count": 40.0,
                "next_10_frame_sequence_parse_valid_rate": 1.0,
                "next_10_frame_sequence_char_accuracy": 0.61,
                "next_10_frame_sequence_changed_map_cell_f1": 0.61,
                "next_10_frame_sequence_exact_match_rate": 0.0,
            },
            baseline_run_id="base",
            trained_run_id="trained",
        )
        watch_report = {
            "run_id": "watch",
            "paired_initial_state_equal": True,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    "fitness_objective_version": "live_rollout_utility_v7",
                    "cumulative_reward": 0.0,
                    "score_delta": 0.0,
                    "depth_max": 1,
                    "depth_delta": 0.0,
                },
                "current": {
                    "fitness_objective_version": "live_rollout_utility_v7",
                    "cumulative_reward": 0.0,
                    "score_delta": 0.0,
                    "depth_max": 1,
                    "depth_delta": 0.0,
                },
                "deltas": {
                    "fitness_score": 0.96,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": -0.36,
                    "bad_message_rate": -0.36,
                    "non_advancing_step_rate": -0.1,
                    "action_repeat_rate": -0.06,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 0.0,
                    "score_delta": 0.0,
                    "cumulative_reward": 0.0,
                    "depth_delta": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=score_report,
            watch_report=watch_report,
        )
        failed_names = {
            requirement["name"]
            for requirement in report["requirements"]
            if requirement["status"] == "failed"
        }

        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "failed")
        self.assertIn("exact_match_rate", failed_names)
        self.assertIn("next_10_frame_sequence_changed_map_cell_f1", failed_names)
        self.assertIn("watch_current_score_or_depth_progress", failed_names)
        self.assertIn("watch_score_or_depth_progress", failed_names)

    def test_training_proof_gate_rejects_relative_gain_without_current_progress(
        self,
    ) -> None:
        watch_report = {
            "schema_version": "learn-nethack.compare-watch-sweep-report.v1",
            "run_id": "watch-sweep",
            "seed_count": 16,
            "paired_initial_state_equal_count": 16,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    **_clean_watch_side(
                        fitness_score=-4.0,
                        cumulative_reward=-1.0,
                        score_delta=0.0,
                        depth_delta=0.0,
                    )
                },
                "current": {
                    **_clean_watch_side(
                        fitness_score=0.5,
                        cumulative_reward=0.0,
                        score_delta=0.0,
                        depth_delta=0.0,
                        zero_progress_episode=1.0,
                    )
                },
                "deltas": {
                    "fitness_score": 4.5,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": -0.1,
                    "bad_message_rate": -0.1,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": 0.0,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 0.0,
                    "score_delta": 0.0,
                    "cumulative_reward": 1.0,
                    "depth_delta": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=_passing_policy_and_next_frame_score_report(),
            watch_report=watch_report,
        )
        failed_names = {
            requirement["name"]
            for requirement in report["requirements"]
            if requirement["status"] == "failed"
        }

        self.assertFalse(report["passed"])
        self.assertIn("watch_current_score_or_depth_progress", failed_names)

    def test_training_proof_gate_failure_takes_precedence_over_missing_v2_fields(
        self,
    ) -> None:
        score_report = build_score_to_beat_report(
            baseline_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.30,
                "next_frame_parse_valid_rate": 1.0,
            },
            trained_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.31,
                "next_frame_parse_valid_rate": 1.0,
            },
            baseline_run_id="base",
            trained_run_id="trained",
        )
        watch_report = {
            "run_id": "old-watch",
            "paired_initial_state_equal": False,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    "cumulative_reward": 0.0,
                    "depth_max": 1,
                },
                "current": {
                    "cumulative_reward": 0.0,
                    "depth_max": 1,
                },
                "deltas": {
                    "fitness_score": -1.0,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": 1.0,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": 0.0,
                    "cumulative_reward": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=score_report,
            watch_report=watch_report,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "failed")

    def test_training_proof_gate_rejects_mismatched_watch_initial_states(
        self,
    ) -> None:
        score_report = build_score_to_beat_report(
            baseline_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.30,
                "next_frame_parse_valid_rate": 1.0,
            },
            trained_metrics={
                "parse_valid_rate": 1.0,
                "action_space_valid_rate": 1.0,
                "exact_match_rate": 0.31,
                "next_frame_parse_valid_rate": 1.0,
            },
            baseline_run_id="base",
            trained_run_id="trained",
        )
        watch_report = {
            "run_id": "watch",
            "paired_initial_state_equal": False,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    **_clean_watch_side(
                        fitness_score=1.0,
                        cumulative_reward=0.0,
                        score_delta=0.0,
                    )
                },
                "current": {**_clean_watch_side()},
                "deltas": {
                    "fitness_score": 1.0,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": 0.0,
                    "bad_message_rate": 0.0,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": 0.0,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 0.0,
                    "score_delta": 1.0,
                    "cumulative_reward": 1.0,
                    "depth_delta": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=score_report,
            watch_report=watch_report,
        )
        failed_names = {
            requirement["name"]
            for requirement in report["requirements"]
            if requirement["status"] == "failed"
        }

        self.assertFalse(report["passed"])
        self.assertIn("watch_paired_initial_state_equal", failed_names)

    def test_training_proof_gate_accepts_sweep_reports_but_rejects_bad_live_terms(
        self,
    ) -> None:
        watch_report = {
            "schema_version": "learn-nethack.compare-watch-sweep-report.v1",
            "run_id": "watch-sweep",
            "seed_count": 16,
            "paired_initial_state_equal_count": 16,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    "fitness_objective_version": "live_rollout_utility_v7",
                    "cumulative_reward": 0.0,
                    "score_delta": 0.0,
                    "depth_delta": 0.0,
                    "depth_max": 1,
                },
                "current": {
                    "fitness_objective_version": "live_rollout_utility_v7",
                    "cumulative_reward": 0.0,
                    "score_delta": 0.0,
                    "depth_delta": 0.0,
                    "depth_max": 1,
                },
                "deltas": {
                    "fitness_score": 0.30,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": 0.16,
                    "bad_message_rate": 0.16,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": 0.16,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 0.0,
                    "score_delta": 0.0,
                    "cumulative_reward": 0.0,
                    "depth_delta": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=_passing_policy_and_next_frame_score_report(),
            watch_report=watch_report,
        )
        failed_names = {
            requirement["name"]
            for requirement in report["requirements"]
            if requirement["status"] == "failed"
        }
        paired_requirement = next(
            requirement
            for requirement in report["requirements"]
            if requirement["name"] == "watch_paired_initial_state_equal"
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(paired_requirement["status"], "passed")
        self.assertIn("watch_wall_message_rate", failed_names)
        self.assertIn("watch_action_repeat_rate", failed_names)
        self.assertIn("watch_score_or_depth_progress", failed_names)

    def test_training_proof_gate_rejects_too_few_sweep_seeds(self) -> None:
        watch_report = {
            "schema_version": "learn-nethack.compare-watch-sweep-report.v1",
            "run_id": "watch-sweep",
            "seed_count": 3,
            "paired_initial_state_equal_count": 3,
            "status": "completed",
            "rollout_metrics": {
                "baseline": {
                    "fitness_objective_version": "live_rollout_utility_v7",
                    "cumulative_reward": 0.0,
                    "score_delta": 0.0,
                    "depth_delta": 0.0,
                    "depth_max": 1,
                },
                "current": {
                    "fitness_objective_version": "live_rollout_utility_v7",
                    "cumulative_reward": 1.0,
                    "score_delta": 1.0,
                    "depth_delta": 0.0,
                    "depth_max": 1,
                },
                "deltas": {
                    "fitness_score": 1.0,
                    "hp_damage_observed": 0.0,
                    "wall_message_rate": -0.1,
                    "bad_message_rate": -0.1,
                    "non_advancing_step_rate": 0.0,
                    "action_repeat_rate": -0.1,
                    "starvation_or_faint_count": 0.0,
                    "menu_or_prompt_step_rate": 0.0,
                    "stuck_menu_or_prompt_loop_count": 0.0,
                    "dirty_live_progress_event_count": 0.0,
                    "score_delta": 1.0,
                    "cumulative_reward": 1.0,
                    "depth_delta": 0.0,
                    "depth_max": 0.0,
                },
            },
        }

        report = build_training_proof_gate_report(
            score_to_beat_report=_passing_policy_and_next_frame_score_report(),
            watch_report=watch_report,
        )
        failed_names = {
            requirement["name"]
            for requirement in report["requirements"]
            if requirement["status"] == "failed"
        }

        self.assertFalse(report["passed"])
        self.assertIn("watch_seed_count", failed_names)

    def test_cli_writes_training_proof_gate_report(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            score_path = root / "score_to_beat.json"
            watch_path = root / "watch_report.json"
            out_path = root / "proof_gate.json"
            score_path.write_text(
                json.dumps(
                    build_score_to_beat_report(
                        baseline_metrics={
                            "parse_valid_rate": 1.0,
                            "action_space_valid_rate": 1.0,
                            "exact_match_rate": 0.30,
                        },
                        trained_metrics={
                            "parse_valid_rate": 1.0,
                            "action_space_valid_rate": 1.0,
                            "exact_match_rate": 0.31,
                        },
                        baseline_run_id="base",
                        trained_run_id="trained",
                    )
                ),
                encoding="utf-8",
            )
            watch_path.write_text(
                json.dumps(
                    {
                        "run_id": "watch",
                        "status": "completed",
                        "rollout_metrics": {"deltas": {}},
                    }
                ),
                encoding="utf-8",
            )

            result = runner.invoke(
                app,
                [
                    "sft",
                    "proof-gate",
                    "--score-to-beat",
                    str(score_path),
                    "--watch-report",
                    str(watch_path),
                    "--out",
                    str(out_path),
                ],
            )
            written = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            written["schema_version"],
            "learn-nethack.training-proof-gate.v1",
        )
        self.assertEqual(written["verdict"], "unproven")


if __name__ == "__main__":
    unittest.main()
