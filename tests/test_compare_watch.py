from __future__ import annotations

import builtins
import json
import sys
import tomllib
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from learn_nethack.action_manifest import ActionEntry, ActionManifest
from learn_nethack.cli import app
from learn_nethack.compare_watch import (
    FITNESS_OBJECTIVE_VERSION,
    ModelWatchSpec,
    TransformerCandidatePolicy,
    build_policy_observation_with_feedback,
    build_policy_messages,
    format_policy_feedback_history,
    format_action_candidate,
    make_nle_env,
    parse_character_list,
    parse_seed_list,
    run_side_by_side_rollout,
    run_side_by_side_rollout_sweep,
    summarize_action_sequence_similarity,
    seed_nle_env,
    select_action_id,
    summarize_rollout_events,
    validate_action_manifest_env_id,
    validate_action_manifest_for_env,
)


ROOT = Path(__file__).resolve().parents[1]


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


class ScriptedPolicy:
    def __init__(self, preferred_action_id: int):
        self.preferred_action_id = preferred_action_id
        self.seen_candidate_sets: list[list[int]] = []
        self.user_prompts: list[str] = []

    def score_actions(
        self,
        *,
        user_prompt: str,
        valid_action_ids: list[int],
    ) -> dict[int, float]:
        self.user_prompts.append(user_prompt)
        self.seen_candidate_sets.append(list(valid_action_ids))
        return {
            action_id: (10.0 if action_id == self.preferred_action_id else 0.0)
            for action_id in valid_action_ids
        }


class FakeEnv:
    def __init__(self, label: str):
        self.label = label
        self.actions: list[int] = []

    def reset(self, *, seed: int):
        blstats = [0] * 13
        blstats[10] = 12
        blstats[12] = 1
        return (
            {
                "tty_chars": [[64, 46], [46, 46]],
                "message": [ord(char) for char in f"{self.label} seed {seed}"],
                "blstats": blstats,
            },
            {},
        )

    def step(self, action_id: int):
        self.actions.append(action_id)
        done = len(self.actions) >= 2
        return (
            {
                "tty_chars": [[46, 64], [46, 46]],
                "message": [ord(char) for char in f"{self.label} action {action_id}"],
                "blstats": [len(self.actions), 10, 1, 0],
            },
            float(action_id + 1),
            done,
            False,
            {
                "message": f"{self.label} action {action_id}",
                "hp": 12 - len(self.actions),
                "depth": 1,
                "hunger": "not hungry",
                "menu_open": False,
                "game_time_advanced": True,
            },
        )

    def close(self) -> None:
        pass


class SmallActionSpaceEnv(FakeEnv):
    def __init__(self, label: str):
        super().__init__(label)
        self.actions = tuple(range(1))


class MatchingActionSpaceEnv(FakeEnv):
    @property
    def unwrapped(self):
        return self

    def __init__(self, label: str = "matching"):
        super().__init__(label)
        self.actions = (107, 106)


class SeedableEnv(FakeEnv):
    @property
    def unwrapped(self):
        return self

    def seed(self, *, core, disp, reseed, lgen):
        self.seed_args = {
            "core": core,
            "disp": disp,
            "reseed": reseed,
            "lgen": lgen,
        }


class CountingBatchPolicy(TransformerCandidatePolicy):
    def __init__(self, *, candidate_batch_size: int = 32) -> None:
        super().__init__(
            ModelWatchSpec(
                role="current",
                model_name="fixture-model",
                adapter_checkpoint=None,
                candidate_batch_size=candidate_batch_size,
            )
        )
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    def _load(self):
        return object(), object(), object()

    def _score_completion_batch(
        self,
        *,
        model,
        tokenizer,
        torch,
        prompt: str,
        completions: list[str],
    ) -> list[float]:
        del model, tokenizer, torch, prompt
        self.batch_calls += 1
        self.batch_sizes.append(len(completions))
        return [float(index) for index, _completion in enumerate(completions)]


class EmptyCacheCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorchWithCuda:
    def __init__(self) -> None:
        self.cuda = EmptyCacheCuda()


class CacheClearingPolicy(TransformerCandidatePolicy):
    def __init__(self) -> None:
        super().__init__(
            ModelWatchSpec(
                role="current",
                model_name="fixture-model",
                adapter_checkpoint=None,
            )
        )
        self.fake_torch = FakeTorchWithCuda()

    def _load(self):
        return object(), object(), self.fake_torch

    def _score_completion_batch(
        self,
        *,
        model,
        tokenizer,
        torch,
        prompt: str,
        completions: list[str],
    ) -> list[float]:
        del model, tokenizer, torch, prompt
        return [float(index) for index, _completion in enumerate(completions)]


class CompareWatchTests(unittest.TestCase):
    def test_action_manifest_matches_live_env_ids_and_raw_key_order(self) -> None:
        validate_action_manifest_for_env(
            _manifest(),
            env_id="NetHackChallenge-v0",
            env=MatchingActionSpaceEnv(),
        )

    def test_action_manifest_rejects_environment_id_mismatch(self) -> None:
        validate_action_manifest_env_id(
            _manifest(),
            env_id="NetHackPairedChallenge-v0",
        )
        with self.assertRaisesRegex(ValueError, "environment mismatch"):
            validate_action_manifest_env_id(
                _manifest(),
                env_id="NetHack-v0",
            )

    def test_action_manifest_rejects_live_action_space_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "id sequence"):
            validate_action_manifest_for_env(
                _manifest(),
                env_id="NetHackChallenge-v0",
                env=SmallActionSpaceEnv("small"),
            )

        wrong_raw_key_order = ActionManifest(
            env_id="NetHackChallenge-v0",
            entries=(
                ActionEntry(
                    action_id=0,
                    nle_action_name="CompassDirection.N",
                    raw_key_code=106,
                    key_label="j",
                ),
                ActionEntry(
                    action_id=1,
                    nle_action_name="CompassDirection.S",
                    raw_key_code=107,
                    key_label="k",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "raw key ordering"):
            validate_action_manifest_for_env(
                wrong_raw_key_order,
                env_id="NetHackChallenge-v0",
                env=MatchingActionSpaceEnv(),
            )

    def test_watch_extra_contains_runtime_deps_without_training_stack(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

        watch_deps = pyproject["project"]["optional-dependencies"]["watch"]

        self.assertIn("gymnasium>=0.29", watch_deps)
        self.assertIn("nle>=1.3.0", watch_deps)
        self.assertIn("torch>=2.3", watch_deps)
        self.assertIn("transformers>=4.42", watch_deps)
        self.assertIn("peft>=0.11", watch_deps)
        self.assertNotIn("unsloth", watch_deps)
        self.assertNotIn("trl>=0.9", watch_deps)

    def test_policy_prompt_and_candidates_preserve_action_json_contract(self) -> None:
        messages = build_policy_messages(
            user_prompt=(
                "Allowed action_ids: [0, 1]\n"
                "Current observation:\nMAP:\n@.\nMESSAGE:\nhello"
            ),
        )

        self.assertEqual(
            messages[0]["content"],
            'You control NetHack through NLE. Return only JSON: {"action_id": int}.',
        )
        self.assertIn("Allowed action_ids: [0, 1]", messages[1]["content"])
        self.assertIn("Current observation:", messages[1]["content"])
        self.assertEqual(format_action_candidate(1), '{"action_id": 1}')

    def test_model_watch_spec_rejects_invalid_context_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown SFT context mode"):
            ModelWatchSpec(
                role="current",
                model_name="model",
                adapter_checkpoint=None,
                context_mode="not-a-mode",
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            ModelWatchSpec(
                role="current",
                model_name="model",
                adapter_checkpoint=None,
                context_token_budget=0,
            )

    def test_policy_feedback_history_renders_nle_outcomes(self) -> None:
        text = format_policy_feedback_history(
            [
                {
                    "action_id": 10,
                    "reward": 0.0,
                    "cumulative_reward": 0.0,
                    "hp": 12,
                    "depth": 1,
                    "game_time_advanced": False,
                    "message": "It's a wall.",
                }
            ]
        )

        self.assertIn("Recent action feedback:", text)
        self.assertIn("action_id=10", text)
        self.assertIn("advanced=False", text)
        self.assertIn('message="It\'s a wall."', text)

        observation = build_policy_observation_with_feedback(
            observation_text="MAP:\n@",
            feedback_history=[],
        )
        self.assertEqual(observation, "MAP:\n@")

        observation = build_policy_observation_with_feedback(
            observation_text="MAP:\n@",
            feedback_history=[
                {
                    "action_id": 10,
                    "reward": 0.0,
                    "cumulative_reward": 0.0,
                    "hp": 12,
                    "depth": 1,
                    "game_time_advanced": False,
                    "message": "It's a wall.",
                }
            ],
        )
        self.assertIn("Recent action feedback:", observation)
        self.assertIn("Current rendered observation:", observation)
        self.assertTrue(observation.endswith("MAP:\n@"))

    def test_select_action_id_only_chooses_from_valid_candidates(self) -> None:
        selected = select_action_id(
            scores_by_action_id={0: 1.0, 1: 2.0, 99: 100.0},
            valid_action_ids=[0, 1],
        )

        self.assertEqual(selected, 1)

        with self.assertRaisesRegex(ValueError, "no scores for valid action ids"):
            select_action_id(scores_by_action_id={99: 1.0}, valid_action_ids=[0, 1])

    def test_transformer_policy_scores_candidates_in_one_batch(self) -> None:
        policy = CountingBatchPolicy()

        scores = policy.score_actions(
            user_prompt="Allowed action_ids: [0, 1, 2]\nCurrent observation:\nMAP:\n@",
            valid_action_ids=[0, 1, 2],
        )

        self.assertEqual(policy.batch_calls, 1)
        self.assertEqual(scores, {0: 0.0, 1: 1.0, 2: 2.0})

    def test_transformer_policy_bounds_candidate_scoring_batch_size(self) -> None:
        policy = CountingBatchPolicy(candidate_batch_size=2)

        policy.score_actions(
            user_prompt="Allowed action_ids: [0, 1, 2, 3, 4]",
            valid_action_ids=[0, 1, 2, 3, 4],
        )

        self.assertEqual(policy.batch_sizes, [2, 2, 1])

    def test_transformer_policy_releases_cuda_cache_after_candidate_scoring(
        self,
    ) -> None:
        policy = CacheClearingPolicy()

        policy.score_actions(
            user_prompt="Allowed action_ids: [0, 1]\nCurrent observation:\nMAP:\n@",
            valid_action_ids=[0, 1],
        )

        self.assertEqual(policy.fake_torch.cuda.empty_cache_calls, 1)

    def test_side_by_side_rollout_writes_events_report_and_viewer(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            current_policy = ScriptedPolicy(preferred_action_id=1)
            baseline_policy = ScriptedPolicy(preferred_action_id=0)
            report = run_side_by_side_rollout(
                run_id="compare-smoke",
                current_spec=ModelWatchSpec(
                    role="current",
                    model_name="google/gemma-4-E4b-it",
                    adapter_checkpoint="artifacts/runs/current-adapter",
                ),
                baseline_spec=ModelWatchSpec(
                    role="baseline",
                    model_name="google/gemma-4-E4b-it",
                    adapter_checkpoint=None,
                ),
                current_policy=current_policy,
                baseline_policy=baseline_policy,
                current_env=FakeEnv("current"),
                baseline_env=FakeEnv("baseline"),
                action_manifest=_manifest(),
                out_dir=out_dir,
                seed=123,
                max_steps=2,
            )

            events_path = out_dir / "events.jsonl"
            html_path = out_dir / "index.html"
            report_path = out_dir / "report.json"
            current_ttyrec_path = out_dir / "current.ttyrec"
            baseline_ttyrec_path = out_dir / "baseline.ttyrec"

            self.assertTrue(events_path.exists())
            self.assertTrue(html_path.exists())
            self.assertTrue(report_path.exists())
            self.assertGreater(current_ttyrec_path.stat().st_size, 0)
            self.assertGreater(baseline_ttyrec_path.stat().st_size, 0)
            self.assertEqual(report["event_count"], 2)
            self.assertEqual(report["viewer_path"], str(html_path))
            self.assertEqual(
                report["current_ttyrec_path"],
                str(current_ttyrec_path),
            )
            self.assertFalse(report["paired_initial_state_equal"])
            self.assertEqual(
                report["rollout_metrics"]["current"]["cumulative_reward"],
                4.0,
            )
            self.assertEqual(
                report["rollout_metrics"]["current"]["hp_damage_observed"],
                2,
            )
            self.assertEqual(
                report["rollout_metrics"]["current"]["action_histogram"],
                {"1": 2},
            )
            self.assertEqual(
                report["rollout_metrics"]["current"]["message_histogram"],
                {"current action 1": 2},
            )
            self.assertEqual(
                report["rollout_metrics"]["current"]["wall_message_count"],
                0,
            )
            self.assertEqual(
                report["rollout_metrics"]["current"]["non_advancing_step_count"],
                0,
            )
            self.assertIn(
                "fitness_score",
                report["rollout_metrics"]["current"],
            )
            self.assertEqual(
                report["rollout_metrics"]["current"]["fitness_objective_version"],
                FITNESS_OBJECTIVE_VERSION,
            )
            self.assertLess(
                report["rollout_metrics"]["current"]["fitness_score"],
                10.0,
            )
            self.assertEqual(
                report["rollout_metrics"]["deltas"]["cumulative_reward"],
                2.0,
            )
            self.assertIn("fitness_score", report["rollout_metrics"]["deltas"])
            self.assertIn("depth_delta", report["rollout_metrics"]["deltas"])

            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            events[0]["schema_version"], "learn-nethack.compare-watch-event.v1"
        )
        self.assertEqual(events[0]["run_id"], "compare-smoke")
        self.assertEqual(events[0]["current"]["action_id"], 1)
        self.assertEqual(events[0]["baseline"]["action_id"], 0)
        self.assertEqual(events[1]["current"]["cumulative_reward"], 4.0)
        self.assertEqual(events[1]["baseline"]["cumulative_reward"], 2.0)
        self.assertEqual(
            events[0]["current"]["prompt_terminal_frame"],
            "MAP:\n@.\n..\nMESSAGE:\ncurrent seed 123\nBLSTATS:\n[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 1]\nINVENTORY:\n<missing>",
        )
        self.assertEqual(
            events[0]["baseline"]["prompt_terminal_frame"],
            "MAP:\n@.\n..\nMESSAGE:\nbaseline seed 123\nBLSTATS:\n[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 1]\nINVENTORY:\n<missing>",
        )
        self.assertEqual(
            events[0]["current"]["policy_observation_text"],
            events[0]["current"]["prompt_terminal_frame"],
        )
        self.assertEqual(events[0]["current"]["policy_feedback_length"], 0)
        self.assertEqual(events[1]["current"]["policy_feedback_length"], 1)
        self.assertIn(
            "current action 1",
            events[1]["current"]["policy_feedback"][0]["message"],
        )
        self.assertNotIn("Recent action feedback:", current_policy.user_prompts[0])
        self.assertIn("Recent action feedback:", current_policy.user_prompts[1])
        self.assertIn("action_id=1", current_policy.user_prompts[1])
        self.assertEqual(current_policy.seen_candidate_sets, [[0, 1], [0, 1]])
        self.assertEqual(baseline_policy.seen_candidate_sets, [[0, 1], [0, 1]])

    def test_single_frame_rollout_never_injects_history(self) -> None:
        with TemporaryDirectory() as tmp:
            policy = ScriptedPolicy(preferred_action_id=1)
            run_side_by_side_rollout(
                run_id="single-frame-contract",
                current_spec=ModelWatchSpec(
                    role="current",
                    model_name="model",
                    adapter_checkpoint="adapter",
                    context_mode="single_frame",
                ),
                baseline_spec=ModelWatchSpec(
                    role="baseline",
                    model_name="model",
                    adapter_checkpoint=None,
                    context_mode="single_frame",
                ),
                current_policy=policy,
                baseline_policy=ScriptedPolicy(preferred_action_id=0),
                current_env=FakeEnv("current"),
                baseline_env=FakeEnv("baseline"),
                action_manifest=_manifest(),
                out_dir=Path(tmp),
                seed=123,
                max_steps=2,
            )

        self.assertEqual(len(policy.user_prompts), 2)
        self.assertNotIn("Recent history:", policy.user_prompts[1])
        self.assertNotIn("Recent action feedback:", policy.user_prompts[1])
        self.assertTrue(policy.user_prompts[1].startswith("Allowed action_ids:"))

    def test_growing_context_rollout_matches_sft_history_shape(self) -> None:
        with TemporaryDirectory() as tmp:
            policy = ScriptedPolicy(preferred_action_id=1)
            run_side_by_side_rollout(
                run_id="growing-context-contract",
                current_spec=ModelWatchSpec(
                    role="current",
                    model_name="model",
                    adapter_checkpoint="adapter",
                    context_mode="growing_context",
                    context_token_budget=512,
                ),
                baseline_spec=ModelWatchSpec(
                    role="baseline",
                    model_name="model",
                    adapter_checkpoint=None,
                    context_mode="growing_context",
                    context_token_budget=512,
                ),
                current_policy=policy,
                baseline_policy=ScriptedPolicy(preferred_action_id=0),
                current_env=FakeEnv("current"),
                baseline_env=FakeEnv("baseline"),
                action_manifest=_manifest(),
                out_dir=Path(tmp),
                seed=123,
                max_steps=2,
            )

        second_prompt = policy.user_prompts[1]
        self.assertIn("Recent history:\n", second_prompt)
        self.assertIn("Previous action_id: 1", second_prompt)
        self.assertNotIn("Recent action feedback:", second_prompt)
        self.assertIn("Current observation:\nMAP:", second_prompt)

    def test_parse_seed_list_rejects_empty_or_invalid_values(self) -> None:
        self.assertEqual(parse_seed_list("101, 202,303"), [101, 202, 303])

        with self.assertRaisesRegex(ValueError, "at least one seed"):
            parse_seed_list(" , ")

    def test_action_sequence_similarity_detects_cross_episode_replay(self) -> None:
        repeated = summarize_action_sequence_similarity(
            [[38, 38, 20], [38, 38, 20], [38, 38, 20], [38, 38, 20]]
        )
        distinct = summarize_action_sequence_similarity([[1, 2], [2, 1]])

        self.assertEqual(
            repeated["cross_episode_action_sequence_pair_count"],
            6,
        )
        self.assertEqual(
            repeated["cross_episode_action_sequence_identity_rate"],
            1.0,
        )
        self.assertEqual(
            repeated["cross_episode_action_sequence_hamming_similarity"],
            1.0,
        )
        self.assertEqual(
            distinct["cross_episode_action_sequence_identity_rate"],
            0.0,
        )
        self.assertEqual(
            distinct["cross_episode_action_sequence_hamming_similarity"],
            0.0,
        )
        with self.assertRaisesRegex(ValueError, "invalid seed"):
            parse_seed_list("101,nope")

    def test_parse_character_list_broadcasts_or_requires_one_per_seed(self) -> None:
        self.assertEqual(
            parse_character_list("mon-hum-neu-mal", episode_count=2),
            ["mon-hum-neu-mal", "mon-hum-neu-mal"],
        )
        self.assertEqual(
            parse_character_list("arc-hum-law-mal,wiz-elf-cha-mal", episode_count=2),
            ["arc-hum-law-mal", "wiz-elf-cha-mal"],
        )
        with self.assertRaisesRegex(ValueError, "match the seed count"):
            parse_character_list(
                "arc-hum-law-mal,mon-hum-neu-mal,wiz-elf-cha-mal",
                episode_count=2,
            )

    def test_side_by_side_sweep_reuses_policies_and_writes_aggregate_report(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            current_policy = ScriptedPolicy(preferred_action_id=1)
            baseline_policy = ScriptedPolicy(preferred_action_id=0)
            emitted_events: list[dict] = []

            report = run_side_by_side_rollout_sweep(
                run_id="compare-sweep",
                current_spec=ModelWatchSpec(
                    role="current",
                    model_name="google/gemma-4-E4b-it",
                    adapter_checkpoint="artifacts/runs/current-adapter",
                ),
                baseline_spec=ModelWatchSpec(
                    role="baseline",
                    model_name="google/gemma-4-E4b-it",
                    adapter_checkpoint=None,
                ),
                current_policy=current_policy,
                baseline_policy=baseline_policy,
                make_current_env=lambda _character: FakeEnv("shared"),
                make_baseline_env=lambda _character: FakeEnv("shared"),
                action_manifest=_manifest(),
                out_dir=out_dir,
                seeds=[101, 202],
                max_steps=2,
                characters="arc-hum-law-mal,wiz-elf-cha-mal",
                event_callback=emitted_events.append,
            )

            report_path = out_dir / "sweep_report.json"

            self.assertTrue(report_path.exists())
            self.assertTrue((out_dir / "seed-101" / "report.json").exists())
            self.assertTrue((out_dir / "seed-202" / "events.jsonl").exists())
            self.assertEqual(
                report["schema_version"],
                "learn-nethack.compare-watch-sweep-report.v1",
            )
            self.assertEqual(report["seed_count"], 2)
            self.assertEqual(report["unique_character_count"], 2)
            self.assertEqual(
                [item["character"] for item in report["seed_reports"]],
                ["arc-hum-law-mal", "wiz-elf-cha-mal"],
            )
            self.assertEqual(report["paired_initial_state_equal_count"], 2)
            self.assertEqual(report["total_event_count"], 4)
            self.assertEqual(len(emitted_events), 4)
            self.assertEqual(emitted_events[0]["character"], "arc-hum-law-mal")
            self.assertEqual(
                report["rollout_metrics"]["deltas"]["cumulative_reward"], 2.0
            )
            self.assertEqual(
                report["rollout_metrics"]["current"]["fitness_objective_version"],
                FITNESS_OBJECTIVE_VERSION,
            )

        self.assertEqual(len(current_policy.seen_candidate_sets), 4)
        self.assertEqual(len(baseline_policy.seen_candidate_sets), 4)

    def test_rollout_summary_counts_wall_messages_and_non_advancing_steps(self) -> None:
        summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "action_id": 3,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "It's a wall.",
                        "game_time_advanced": False,
                    }
                },
                {
                    "current": {
                        "action_id": 3,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "It's solid stone.",
                        "game_time_advanced": True,
                    }
                },
            ],
            side="current",
            reset_observation={"blstats": [0] * 10 + [12]},
        )

        self.assertEqual(summary["action_histogram"], {"3": 2})
        self.assertEqual(
            summary["message_histogram"],
            {"It's a wall.": 1, "It's solid stone.": 1},
        )
        self.assertEqual(summary["wall_message_count"], 2)
        self.assertEqual(summary["non_advancing_step_count"], 1)
        self.assertEqual(summary["action_mode_count"], 2)
        self.assertEqual(summary["action_repeat_rate"], 1.0)
        self.assertEqual(summary["action_collapse_excess"], 1)
        self.assertLess(summary["fitness_score"], 0.0)
        self.assertEqual(
            summary["fitness_components"]["non_advancing_step_penalty"],
            -0.2,
        )
        self.assertEqual(
            summary["fitness_components"]["action_collapse_penalty"],
            -0.05,
        )
        self.assertEqual(
            summary["fitness_components"]["wall_message_penalty"],
            -0.2,
        )

    def test_rollout_summary_uses_live_utility_components_from_ml_analysis(
        self,
    ) -> None:
        reset_observation = {
            "tty_chars": [[64, 46], [46, 46]],
            "message": [],
            "blstats": [0] * 27,
        }
        reset_observation["blstats"][9] = 0
        reset_observation["blstats"][10] = 12
        reset_observation["blstats"][12] = 1
        summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "terminal_frame": ("MAP:\n.@\n..\nMESSAGE:\nYou move.\n"),
                        "action_id": 1,
                        "reward": 1.0,
                        "cumulative_reward": 1.0,
                        "score": 5,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "You move.",
                        "game_time_advanced": True,
                    }
                },
                {
                    "current": {
                        "terminal_frame": ("MAP:\n..@\n..<\nMESSAGE:\nYou descend.\n"),
                        "action_id": 1,
                        "reward": 2.0,
                        "cumulative_reward": 3.0,
                        "score": 12,
                        "done": False,
                        "hp": 12,
                        "depth": 2,
                        "message": "You descend.",
                        "game_time_advanced": True,
                    }
                },
            ],
            side="current",
            reset_observation=reset_observation,
        )

        self.assertEqual(
            summary["fitness_objective_version"], FITNESS_OBJECTIVE_VERSION
        )
        self.assertEqual(summary["score_delta"], 12)
        self.assertEqual(summary["depth_delta"], 1)
        self.assertGreater(summary["visible_map_novelty_count"], 0)
        self.assertEqual(summary["meaningful_event_count"], 2)
        self.assertGreater(
            summary["fitness_components"]["normalized_score_delta_bonus"],
            0.0,
        )
        self.assertEqual(summary["fitness_components"]["depth_delta_bonus"], 3.0)

    def test_rollout_progress_bonus_requires_clean_progress_event(self) -> None:
        reset_observation = {
            "tty_chars": [[64, 46], [46, 46]],
            "message": [],
            "blstats": [0] * 27,
        }
        reset_observation["blstats"][9] = 0
        reset_observation["blstats"][10] = 12
        reset_observation["blstats"][12] = 1

        summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "terminal_frame": (
                            "MAP:\n@.\nMESSAGE:\nYou cannot move there.\n"
                        ),
                        "action_id": 1,
                        "reward": 1.0,
                        "cumulative_reward": 1.0,
                        "score": 5,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "You cannot move there.",
                        "game_time_advanced": False,
                    }
                },
                {
                    "current": {
                        "terminal_frame": "MAP:\n.@\nMESSAGE:\nYou hit.\n",
                        "action_id": 0,
                        "reward": 1.0,
                        "cumulative_reward": 2.0,
                        "score": 9,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "You hit.",
                        "game_time_advanced": True,
                    }
                },
            ],
            side="current",
            reset_observation=reset_observation,
        )

        self.assertEqual(
            summary["fitness_objective_version"], "live_rollout_utility_v7"
        )
        self.assertEqual(summary["raw_live_progress_event_count"], 2)
        self.assertEqual(summary["clean_live_progress_event_count"], 1)
        self.assertEqual(summary["live_progress_event_count"], 1)
        self.assertEqual(summary["dirty_live_progress_event_count"], 1)
        self.assertEqual(summary["meaningful_event_count"], 1)
        self.assertEqual(
            summary["fitness_components"]["live_progress_event_bonus"],
            0.5,
        )
        self.assertEqual(
            summary["fitness_components"]["dirty_live_progress_event_penalty"],
            -1.0,
        )

    def test_rollout_fitness_rejects_proxy_gain_without_live_progress(self) -> None:
        def frame(width: int, message: str) -> str:
            return f"MAP:\n{'.' * width}@\nMESSAGE:\n{message}\n"

        def event(action_id: int, message: str, width: int) -> dict:
            return {
                "current": {
                    "terminal_frame": frame(width, message),
                    "action_id": action_id,
                    "reward": 0.0,
                    "cumulative_reward": 0.0,
                    "score": 0,
                    "done": False,
                    "hp": 12,
                    "depth": 1,
                    "message": message,
                    "game_time_advanced": True,
                }
            }

        baseline = summarize_rollout_events(
            [
                event(0, "You move.", 1),
                event(1, "You move.", 2),
                event(2, "You move.", 3),
                event(3, "You move.", 4),
                event(4, "You move.", 5),
                event(5, "It's a wall.", 5),
                event(0, "It's a wall.", 5),
                event(1, "You move.", 5),
                event(2, "You move.", 5),
                event(3, "You move.", 5),
            ],
            side="current",
            reset_observation=None,
        )
        current = summarize_rollout_events(
            [
                event(1, "You move.", 1),
                event(1, "You move.", 2),
                event(1, "You move.", 3),
                event(1, "It's a wall.", 3),
                event(1, "You move.", 4),
                event(1, "It's a wall.", 4),
                event(1, "You move.", 5),
                event(1, "It's a wall.", 5),
                event(1, "You move.", 6),
                event(2, "It's a wall.", 6),
            ],
            side="current",
            reset_observation=None,
        )

        self.assertGreater(
            current["visible_map_novelty_count"],
            baseline["visible_map_novelty_count"],
        )
        self.assertGreater(
            current["wall_message_rate"],
            baseline["wall_message_rate"],
        )
        self.assertGreater(
            current["action_repeat_rate"],
            baseline["action_repeat_rate"],
        )
        self.assertLess(current["fitness_score"], baseline["fitness_score"])

    def test_rollout_summary_keeps_visible_novelty_out_of_meaningful_events(
        self,
    ) -> None:
        summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "terminal_frame": "MAP:\n.@.\nMESSAGE:\nYou move.\n",
                        "action_id": 1,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "score": 0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "You move.",
                        "game_time_advanced": True,
                    }
                },
                {
                    "current": {
                        "terminal_frame": "MAP:\n..@\nMESSAGE:\nYou move.\n",
                        "action_id": 1,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "score": 0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "You move.",
                        "game_time_advanced": True,
                    }
                },
            ],
            side="current",
            reset_observation=None,
        )

        self.assertGreater(summary["visible_map_novelty_count"], 0)
        self.assertEqual(summary["meaningful_event_count"], 0)
        self.assertEqual(
            summary["fitness_components"]["visible_map_novelty_bonus"],
            0.0,
        )
        self.assertEqual(
            summary["fitness_components"]["meaningful_event_bonus"],
            0.0,
        )
        self.assertEqual(summary["zero_progress_episode"], 1)
        self.assertLess(summary["fitness_score"], 0.0)

    def test_rollout_fitness_penalizes_bad_messages_and_zero_progress(self) -> None:
        def event(
            *,
            action_id: int,
            message: str,
            width: int,
            advanced: bool = True,
        ) -> dict:
            return {
                "current": {
                    "terminal_frame": f"MAP:\n{'.' * width}@\nMESSAGE:\n{message}\n",
                    "action_id": action_id,
                    "reward": 0.0,
                    "cumulative_reward": 0.0,
                    "score": 0,
                    "done": False,
                    "hp": 12,
                    "depth": 1,
                    "message": message,
                    "game_time_advanced": advanced,
                }
            }

        quiet_zero_progress = summarize_rollout_events(
            [
                event(action_id=0, message="", width=1),
                event(action_id=1, message="", width=1),
                event(action_id=2, message="", width=1),
                event(action_id=3, message="", width=1),
            ],
            side="current",
            reset_observation=None,
        )
        bad_proxy_progress = summarize_rollout_events(
            [
                event(
                    action_id=30,
                    message="What do you want to eat? [fghi or ?*]",
                    width=2,
                    advanced=False,
                ),
                event(
                    action_id=30,
                    message="You cannot eat that!",
                    width=3,
                    advanced=False,
                ),
                event(
                    action_id=30,
                    message="You don't have that object.",
                    width=4,
                    advanced=False,
                ),
                event(
                    action_id=30,
                    message="Never mind.",
                    width=5,
                    advanced=False,
                ),
            ],
            side="current",
            reset_observation=None,
        )

        self.assertEqual(
            bad_proxy_progress["fitness_objective_version"],
            "live_rollout_utility_v7",
        )
        self.assertEqual(bad_proxy_progress["bad_message_count"], 4)
        self.assertEqual(bad_proxy_progress["bad_message_rate"], 1.0)
        self.assertEqual(bad_proxy_progress["zero_progress_episode"], 1)
        self.assertIn(
            "bad_message_rate_penalty",
            bad_proxy_progress["fitness_components"],
        )
        self.assertIn(
            "zero_progress_episode_penalty",
            bad_proxy_progress["fitness_components"],
        )
        self.assertLess(
            bad_proxy_progress["fitness_score"],
            quiet_zero_progress["fitness_score"],
        )

    def test_rollout_summary_detects_frame_only_prompt_and_wall_text(self) -> None:
        summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "terminal_frame": (
                            "MAP:\n"
                            " Extended Commands List\n"
                            " down               go down a staircase\n"
                            " (1 of 5)\n"
                            "MESSAGE:\n<missing>\n"
                        ),
                        "action_id": 30,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "score": 0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "",
                        "game_time_advanced": False,
                    }
                },
                {
                    "current": {
                        "terminal_frame": (
                            "MAP:\nIt's solid stone.\n   --@--\nMESSAGE:\n<missing>\n"
                        ),
                        "action_id": 1,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "score": 0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "",
                        "game_time_advanced": True,
                    }
                },
            ],
            side="current",
            reset_observation={"blstats": [0] * 10 + [12]},
        )

        self.assertEqual(summary["menu_or_prompt_step_count"], 1)
        self.assertEqual(summary["wall_message_count"], 1)
        self.assertEqual(summary["bad_message_count"], 2)
        self.assertLess(summary["fitness_score"], 0.0)

    def test_rollout_summary_does_not_count_message_text_as_map_novelty(
        self,
    ) -> None:
        reset_observation = {
            "tty_chars": [[64, 46], [46, 46]],
            "message": [],
            "blstats": [0] * 27,
        }
        reset_observation["blstats"][10] = 12
        reset_observation["blstats"][12] = 1

        wall_frame = (
            "MAP:\n"
            "It's a wall.\n\n"
            "           ---+--------\n"
            "           |........f.|\n"
            "           |(.........|\n"
            "           |.<.......@|\n"
            "           ------------\n"
            "MESSAGE:\n"
            "It's a wall.\n"
        )
        prompt_frame = (
            "MAP:\n"
            "What do you want to eat? [ghij or ?*]\n\n"
            "           ---+--------\n"
            "           |........f.|\n"
            "           |(.........|\n"
            "           |.<.......@|\n"
            "           ------------\n"
            "MESSAGE:\n"
            "What do you want to eat? [ghij or ?*]\n"
        )
        unchanged_summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "terminal_frame": wall_frame,
                        "action_id": 1,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "It's a wall.",
                        "game_time_advanced": True,
                    }
                },
                {
                    "current": {
                        "terminal_frame": wall_frame,
                        "action_id": 1,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "It's a wall.",
                        "game_time_advanced": True,
                    }
                },
            ],
            side="current",
            reset_observation=reset_observation,
        )
        changed_message_summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "terminal_frame": wall_frame,
                        "action_id": 1,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "It's a wall.",
                        "game_time_advanced": True,
                    }
                },
                {
                    "current": {
                        "terminal_frame": prompt_frame,
                        "action_id": 30,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 12,
                        "depth": 1,
                        "message": "What do you want to eat? [ghij or ?*]",
                        "game_time_advanced": True,
                    }
                },
            ],
            side="current",
            reset_observation=reset_observation,
        )

        self.assertEqual(
            changed_message_summary["visible_map_novelty_count"],
            unchanged_summary["visible_map_novelty_count"],
        )
        self.assertEqual(
            changed_message_summary["meaningful_event_count"],
            unchanged_summary["meaningful_event_count"],
        )

    def test_rollout_summary_penalizes_starvation_and_stuck_prompts(self) -> None:
        summary = summarize_rollout_events(
            [
                {
                    "current": {
                        "terminal_frame": "MAP:\n@.\nMESSAGE:\n--More--\n",
                        "action_id": 7,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 4,
                        "depth": 1,
                        "message": "--More--",
                        "hunger": "fainting from hunger",
                        "menu_open": True,
                        "game_time_advanced": False,
                    }
                },
                {
                    "current": {
                        "terminal_frame": "MAP:\n@.\nMESSAGE:\n--More--\n",
                        "action_id": 7,
                        "reward": 0.0,
                        "cumulative_reward": 0.0,
                        "done": False,
                        "hp": 4,
                        "depth": 1,
                        "message": "--More--",
                        "hunger": "fainting from hunger",
                        "menu_open": True,
                        "game_time_advanced": False,
                    }
                },
            ],
            side="current",
            reset_observation={"blstats": [0] * 10 + [4]},
        )

        self.assertEqual(summary["hunger_warning_count"], 2)
        self.assertEqual(summary["starvation_or_faint_count"], 2)
        self.assertEqual(summary["menu_or_prompt_step_count"], 2)
        self.assertEqual(summary["stuck_menu_or_prompt_loop_count"], 1)
        self.assertEqual(
            summary["fitness_components"]["starvation_or_faint_penalty"],
            -3.0,
        )
        self.assertEqual(
            summary["fitness_components"]["stuck_menu_or_prompt_loop_penalty"],
            -0.5,
        )

    def test_side_by_side_rollout_rejects_manifest_env_action_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "action space has 1 actions"):
                run_side_by_side_rollout(
                    run_id="compare-smoke",
                    current_spec=ModelWatchSpec(
                        role="current",
                        model_name="google/gemma-4-E4b-it",
                        adapter_checkpoint=None,
                    ),
                    baseline_spec=ModelWatchSpec(
                        role="baseline",
                        model_name="google/gemma-4-E4b-it",
                        adapter_checkpoint=None,
                    ),
                    current_policy=ScriptedPolicy(preferred_action_id=0),
                    baseline_policy=ScriptedPolicy(preferred_action_id=0),
                    current_env=SmallActionSpaceEnv("current"),
                    baseline_env=FakeEnv("baseline"),
                    action_manifest=_manifest(),
                    out_dir=Path(tmp),
                    seed=123,
                    max_steps=1,
                )

    def test_cli_writes_compare_watch_contract_without_importing_heavy_deps(
        self,
    ) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "action_manifest.json"
            out_dir = root / "watch"
            _manifest().save(manifest_path)

            result = runner.invoke(
                app,
                [
                    "watch",
                    "compare",
                    "--run-id",
                    "contract-smoke",
                    "--action-manifest",
                    str(manifest_path),
                    "--current-checkpoint",
                    "artifacts/runs/current-adapter",
                    "--out",
                    str(out_dir),
                    "--dry-run-contract",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            contract = json.loads((out_dir / "compare_watch_contract.json").read_text())

        self.assertEqual(
            contract["schema_version"], "learn-nethack.compare-watch-contract.v1"
        )
        self.assertEqual(contract["run_id"], "contract-smoke")
        self.assertEqual(
            contract["current"]["adapter_checkpoint"], "artifacts/runs/current-adapter"
        )
        self.assertIsNone(contract["baseline"]["adapter_checkpoint"])
        self.assertEqual(contract["valid_action_ids"], [0, 1])

    def test_cli_allows_base_gemma_current_without_checkpoint(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "action_manifest.json"
            out_dir = root / "watch"
            _manifest().save(manifest_path)

            result = runner.invoke(
                app,
                [
                    "watch",
                    "compare",
                    "--run-id",
                    "base-only-smoke",
                    "--action-manifest",
                    str(manifest_path),
                    "--model-name",
                    "google/gemma-3-270m-it",
                    "--out",
                    str(out_dir),
                    "--dry-run-contract",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            contract = json.loads((out_dir / "compare_watch_contract.json").read_text())

        self.assertEqual(contract["current"]["model_name"], "google/gemma-3-270m-it")
        self.assertIsNone(contract["current"]["adapter_checkpoint"])
        self.assertIsNone(contract["baseline"]["adapter_checkpoint"])

    def test_cli_writes_compare_watch_sweep_contract_without_heavy_deps(
        self,
    ) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "action_manifest.json"
            out_dir = root / "watch-sweep"
            _manifest().save(manifest_path)

            result = runner.invoke(
                app,
                [
                    "watch",
                    "sweep",
                    "--run-id",
                    "contract-sweep",
                    "--action-manifest",
                    str(manifest_path),
                    "--current-checkpoint",
                    "artifacts/runs/current-adapter",
                    "--out",
                    str(out_dir),
                    "--seeds",
                    "101,202",
                    "--dry-run-contract",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            contract = json.loads(
                (out_dir / "compare_watch_sweep_contract.json").read_text()
            )

        self.assertEqual(
            contract["schema_version"], "learn-nethack.compare-watch-sweep-contract.v1"
        )
        self.assertEqual(contract["run_id"], "contract-sweep")
        self.assertEqual(contract["seeds"], [101, 202])
        self.assertEqual(contract["valid_action_ids"], [0, 1])

    def test_make_nle_env_imports_nle_before_gym_make_and_pins_character(
        self,
    ) -> None:
        fake_gymnasium = types.ModuleType("gymnasium")
        fake_nle = types.ModuleType("nle")
        imported_nle = False

        def fake_make(env_id: str, **kwargs):
            if not imported_nle:
                raise AssertionError("nle must be imported before gym.make")
            return {"env_id": env_id, "kwargs": kwargs}

        fake_gymnasium.make = fake_make
        original_import = builtins.__import__
        original_gymnasium = sys.modules.get("gymnasium")
        original_nle = sys.modules.get("nle")
        sys.modules["gymnasium"] = fake_gymnasium
        sys.modules["nle"] = fake_nle

        def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
            nonlocal imported_nle
            if name == "nle":
                imported_nle = True
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = recording_import
        try:
            self.assertEqual(
                make_nle_env(
                    "NetHackChallenge-v0",
                    character="bar-orc-cha-fem",
                ),
                {
                    "env_id": "NetHackChallenge-v0",
                    "kwargs": {"character": "bar-orc-cha-fem"},
                },
            )
        finally:
            builtins.__import__ = original_import
            if original_gymnasium is None:
                sys.modules.pop("gymnasium", None)
            else:
                sys.modules["gymnasium"] = original_gymnasium
            if original_nle is None:
                sys.modules.pop("nle", None)
            else:
                sys.modules["nle"] = original_nle

    def test_seed_nle_env_sets_all_rngs_when_supported(self) -> None:
        env = SeedableEnv("seedable")

        self.assertTrue(seed_nle_env(env, seed=123))

        self.assertEqual(
            env.seed_args,
            {
                "core": 123,
                "disp": 123,
                "reseed": False,
                "lgen": 123,
            },
        )


if __name__ == "__main__":
    unittest.main()
