"""Side-by-side NetHack watch harness for a checkpoint and base Gemma."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from learn_nethack.action_manifest import ActionManifest, load_action_manifest
from learn_nethack.observations import render_observation_text
from learn_nethack import policy_feedback
from learn_nethack.sft_rows import POLICY_SYSTEM_PROMPT


COMPARE_EVENT_SCHEMA_VERSION = "learn-nethack.compare-watch-event.v1"
COMPARE_REPORT_SCHEMA_VERSION = "learn-nethack.compare-watch-report.v1"
COMPARE_SWEEP_REPORT_SCHEMA_VERSION = "learn-nethack.compare-watch-sweep-report.v1"
COMPARE_CONTRACT_SCHEMA_VERSION = "learn-nethack.compare-watch-contract.v1"
COMPARE_SWEEP_CONTRACT_SCHEMA_VERSION = "learn-nethack.compare-watch-sweep-contract.v1"
DEFAULT_NLE_CHARACTER = "mon-hum-neu-mal"
FITNESS_OBJECTIVE_VERSION = "live_rollout_utility_v7"
ACTION_COLLAPSE_RATE_THRESHOLD = 0.50
MAX_VISIBLE_MAP_NOVELTY_BONUS_COUNT = 5
MAX_MEANINGFUL_EVENT_BONUS_COUNT = 5
FITNESS_WEIGHTS = {
    "normalized_score_delta": 1.0,
    "cumulative_reward": 0.25,
    "depth_delta": 3.0,
    "visible_map_novelty": 0.0,
    "meaningful_event": 0.10,
    "live_progress_event": 0.50,
    "hp_damage": -0.50,
    "death": -8.0,
    "starvation_or_faint": -1.50,
    "wall_or_solid_stone_message": -0.10,
    "wall_message_rate": -2.50,
    "bad_message": -0.10,
    "bad_message_rate": -1.50,
    "non_advancing_step": -0.20,
    "non_advancing_step_rate": -1.00,
    "action_collapse_excess": -0.05,
    "action_collapse_rate_excess": -2.00,
    "menu_or_prompt_step_rate": -1.50,
    "stuck_menu_or_prompt_loop": -0.50,
    "dirty_live_progress_event": -1.00,
    "zero_progress_episode": -3.00,
}
MAP_NOVELTY_CHARS = frozenset('|-+.#@<>$()[]{}%?!/\\"=_*,`^')
MAX_POLICY_FEEDBACK_ITEMS = policy_feedback.MAX_POLICY_FEEDBACK_ITEMS
build_policy_observation_with_feedback = (
    policy_feedback.build_policy_observation_with_feedback
)
format_policy_feedback_history = policy_feedback.format_policy_feedback_history


@dataclass(frozen=True)
class ModelWatchSpec:
    role: str
    model_name: str
    adapter_checkpoint: str | None


class ActionScoringPolicy(Protocol):
    def score_actions(
        self,
        *,
        observation_text: str,
        valid_action_ids: list[int],
    ) -> dict[int, float]:
        """Return one score per candidate action id."""


class NetHackEnv(Protocol):
    def reset(self, *, seed: int): ...

    def step(self, action_id: int): ...

    def close(self) -> None: ...


def validate_action_manifest_env_id(
    action_manifest: ActionManifest,
    *,
    env_id: str,
) -> None:
    """Require the action manifest to name the environment being evaluated."""
    if action_manifest.env_id != env_id:
        raise ValueError(
            "action manifest environment mismatch: "
            f"manifest={action_manifest.env_id!r}, requested={env_id!r}"
        )


def validate_action_manifest_for_env(
    action_manifest: ActionManifest,
    *,
    env_id: str,
    env: NetHackEnv,
) -> None:
    """Require manifest ids and raw key ordering to match the live NLE env."""
    validate_action_manifest_env_id(action_manifest, env_id=env_id)
    target = getattr(env, "unwrapped", env)
    actions = getattr(target, "actions", None)
    if actions is None:
        raise RuntimeError(
            f"NLE environment {env_id!r} does not expose an action sequence"
        )

    expected_action_ids = list(range(len(actions)))
    manifest_action_ids = action_manifest.valid_action_ids()
    if manifest_action_ids != expected_action_ids:
        raise ValueError(
            "action manifest id sequence does not match the live NLE action space: "
            f"manifest={manifest_action_ids}, expected={expected_action_ids}"
        )

    try:
        env_raw_keys = [int(action) for action in actions]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"NLE environment {env_id!r} exposes a non-integer action"
        ) from exc
    manifest_raw_keys = [entry.raw_key_code for entry in action_manifest.entries]
    if manifest_raw_keys != env_raw_keys:
        raise ValueError(
            "action manifest raw key ordering does not match the live NLE action "
            f"space: manifest={manifest_raw_keys}, expected={env_raw_keys}"
        )


def parse_seed_list(seeds: str | Sequence[int]) -> list[int]:
    """Parse a comma-separated seed list for deterministic watch sweeps."""
    if isinstance(seeds, str):
        parts = [part.strip() for part in seeds.split(",")]
        raw_values: Sequence[Any] = [part for part in parts if part]
    else:
        raw_values = list(seeds)
    parsed: list[int] = []
    for value in raw_values:
        try:
            parsed.append(int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid seed: {value!r}") from exc
    if not parsed:
        raise ValueError("at least one seed is required")
    return parsed


def format_action_candidate(action_id: int) -> str:
    """Return the exact policy JSON candidate scored by the harness."""
    return f'{{"action_id": {int(action_id)}}}'


def build_policy_messages(
    *,
    observation_text: str,
    valid_action_ids: Sequence[int],
) -> list[dict[str, str]]:
    """Build the policy prompt used for candidate-action scoring."""
    action_ids = [int(action_id) for action_id in valid_action_ids]
    return [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"Allowed action_ids: {action_ids}",
                    "Current observation:",
                    observation_text,
                ]
            ),
        },
    ]


def select_action_id(
    *,
    scores_by_action_id: Mapping[int, float],
    valid_action_ids: Sequence[int],
) -> int:
    """Choose the highest-scored valid action and ignore out-of-space scores."""
    valid_set = {int(action_id) for action_id in valid_action_ids}
    valid_scores = [
        (int(action_id), float(score))
        for action_id, score in scores_by_action_id.items()
        if int(action_id) in valid_set
    ]
    if not valid_scores:
        raise ValueError("no scores for valid action ids")
    return max(valid_scores, key=lambda item: (item[1], -item[0]))[0]


class TransformerCandidatePolicy:
    """Score exact action JSON candidates with a base model and optional adapter."""

    def __init__(
        self,
        spec: ModelWatchSpec,
        *,
        device: str | None = None,
        torch_dtype: str = "auto",
    ) -> None:
        self.spec = spec
        self.device = device
        self.torch_dtype = torch_dtype
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None

    def score_actions(
        self,
        *,
        observation_text: str,
        valid_action_ids: list[int],
    ) -> dict[int, float]:
        model, tokenizer, torch = self._load()
        messages = build_policy_messages(
            observation_text=observation_text,
            valid_action_ids=valid_action_ids,
        )
        prompt = _apply_chat_template(tokenizer, messages) + '{"action_id": '
        completions = [f"{int(action_id)}}}" for action_id in valid_action_ids]
        try:
            scores = self._score_completion_batch(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                prompt=prompt,
                completions=completions,
            )
        finally:
            _release_torch_cuda_cache(torch)
        return {
            int(action_id): float(score)
            for action_id, score in zip(valid_action_ids, scores, strict=True)
        }

    def _load(self) -> tuple[Any, Any, Any]:
        if (
            self._model is not None
            and self._tokenizer is not None
            and self._torch is not None
        ):
            return self._model, self._tokenizer, self._torch

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - requires model deps.
            raise RuntimeError(
                "torch and transformers are required for compare-watch model scoring; "
                "run with the watch extra or inside the Modal training image"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)
        model_kwargs: dict[str, Any] = {}
        if self.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        if self.device is None:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_name,
            **model_kwargs,
        )
        if self.spec.adapter_checkpoint is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - requires model deps.
                raise RuntimeError(
                    "peft is required to load the current checkpoint adapter"
                ) from exc
            model = PeftModel.from_pretrained(model, self.spec.adapter_checkpoint)
        if self.device is not None:
            model = model.to(self.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        return model, tokenizer, torch

    def _score_completion_batch(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch: Any,
        prompt: str,
        completions: list[str],
    ) -> list[float]:
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        completion_id_rows = [
            tokenizer(
                completion,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            for completion in completions
        ]
        pad_token_id = _pad_token_id(tokenizer)
        max_completion_length = max(int(row.shape[0]) for row in completion_id_rows)
        completion_ids = torch.full(
            (len(completion_id_rows), max_completion_length),
            fill_value=pad_token_id,
            dtype=prompt_ids.dtype,
        )
        completion_mask = torch.zeros_like(completion_ids)
        for row_index, row in enumerate(completion_id_rows):
            row_length = int(row.shape[0])
            completion_ids[row_index, :row_length] = row
            completion_mask[row_index, :row_length] = 1
        device = getattr(model, "device", None)
        if device is None:
            device = next(model.parameters()).device
        prompt_attention_mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            prompt_outputs = model(
                input_ids=prompt_ids.to(device),
                attention_mask=prompt_attention_mask.to(device),
                use_cache=True,
            )
        first_token_ids = completion_ids[:, 0].to(device)
        first_log_probs = torch.log_softmax(prompt_outputs.logits[:, -1, :], dim=-1)
        sequence_scores = (
            first_log_probs.repeat(len(completion_id_rows), 1)
            .gather(
                dim=-1,
                index=first_token_ids.unsqueeze(-1),
            )
            .squeeze(-1)
        )
        if max_completion_length == 1:
            return [float(score.item()) for score in sequence_scores]

        past_key_values = prompt_outputs.past_key_values
        past_key_values.batch_repeat_interleave(len(completion_id_rows))
        prefix_input_ids = completion_ids[:, :-1]
        prefix_attention_mask = completion_mask[:, :-1]
        prompt_attention_mask = torch.ones(
            (len(completion_id_rows), int(prompt_ids.shape[1])),
            dtype=prefix_attention_mask.dtype,
        )
        attention_mask = torch.cat(
            [prompt_attention_mask, prefix_attention_mask],
            dim=1,
        )
        with torch.no_grad():
            outputs = model(
                input_ids=prefix_input_ids.to(device),
                attention_mask=attention_mask.to(device),
                past_key_values=past_key_values,
                use_cache=False,
            )
        shifted_labels = completion_ids[:, 1:].to(device)
        target_mask = completion_mask[:, 1:].to(device).bool()
        token_log_probs = torch.log_softmax(outputs.logits, dim=-1)
        selected_log_probs = token_log_probs.gather(
            dim=-1,
            index=shifted_labels.unsqueeze(-1),
        ).squeeze(-1)
        sequence_scores = sequence_scores + (selected_log_probs * target_mask).sum(
            dim=1
        )
        return [float(score.item()) for score in sequence_scores]


def run_checkpoint_compare(
    *,
    run_id: str,
    current_checkpoint: str | None,
    action_manifest_path: str | Path,
    out_dir: str | Path,
    model_name: str = "google/gemma-4-E4b-it",
    env_id: str = "NetHackChallenge-v0",
    character: str = DEFAULT_NLE_CHARACTER,
    seed: int = 20260615,
    max_steps: int = 80,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the current adapter and base Gemma side by side in seeded NLE envs."""
    manifest = load_action_manifest(action_manifest_path)
    current_spec = ModelWatchSpec(
        role="current",
        model_name=model_name,
        adapter_checkpoint=current_checkpoint,
    )
    baseline_spec = ModelWatchSpec(
        role="baseline",
        model_name=model_name,
        adapter_checkpoint=None,
    )
    current_env = make_nle_env(env_id, character=character)
    baseline_env = make_nle_env(env_id, character=character)
    try:
        validate_action_manifest_for_env(
            manifest,
            env_id=env_id,
            env=current_env,
        )
        validate_action_manifest_for_env(
            manifest,
            env_id=env_id,
            env=baseline_env,
        )
        return run_side_by_side_rollout(
            run_id=run_id,
            current_spec=current_spec,
            baseline_spec=baseline_spec,
            current_policy=TransformerCandidatePolicy(current_spec, device=device),
            baseline_policy=TransformerCandidatePolicy(baseline_spec, device=device),
            current_env=current_env,
            baseline_env=baseline_env,
            action_manifest=manifest,
            out_dir=out_dir,
            seed=seed,
            max_steps=max_steps,
        )
    finally:
        current_env.close()
        baseline_env.close()


def run_checkpoint_compare_sweep(
    *,
    run_id: str,
    current_checkpoint: str | None,
    action_manifest_path: str | Path,
    out_dir: str | Path,
    seeds: str | Sequence[int],
    model_name: str = "google/gemma-4-E4b-it",
    env_id: str = "NetHackChallenge-v0",
    character: str = DEFAULT_NLE_CHARACTER,
    max_steps: int = 80,
    device: str | None = None,
) -> dict[str, Any]:
    """Run a multi-seed checkpoint comparison while reusing loaded policies."""
    manifest = load_action_manifest(action_manifest_path)
    validate_action_manifest_env_id(manifest, env_id=env_id)
    probe_env = make_nle_env(env_id, character=character)
    try:
        validate_action_manifest_for_env(
            manifest,
            env_id=env_id,
            env=probe_env,
        )
    finally:
        probe_env.close()
    seed_values = parse_seed_list(seeds)
    current_spec = ModelWatchSpec(
        role="current",
        model_name=model_name,
        adapter_checkpoint=current_checkpoint,
    )
    baseline_spec = ModelWatchSpec(
        role="baseline",
        model_name=model_name,
        adapter_checkpoint=None,
    )
    return run_side_by_side_rollout_sweep(
        run_id=run_id,
        current_spec=current_spec,
        baseline_spec=baseline_spec,
        current_policy=TransformerCandidatePolicy(current_spec, device=device),
        baseline_policy=TransformerCandidatePolicy(baseline_spec, device=device),
        make_current_env=lambda: make_nle_env(env_id, character=character),
        make_baseline_env=lambda: make_nle_env(env_id, character=character),
        action_manifest=manifest,
        out_dir=out_dir,
        seeds=seed_values,
        max_steps=max_steps,
    )


def make_nle_env(
    env_id: str,
    *,
    character: str = DEFAULT_NLE_CHARACTER,
) -> NetHackEnv:
    """Create an NLE environment without making NLE a package import dependency."""
    try:
        import nle  # noqa: F401
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - depends on optional local-nle.
        raise RuntimeError(
            "gymnasium and nle are required to run compare-watch rollouts; "
            "run with `uv run --extra local-nle ...` or inside the Modal image"
        ) from exc
    try:
        return gym.make(env_id, character=character)
    except Exception as exc:  # pragma: no cover - depends on local NLE registry.
        raise RuntimeError(f"failed to create NLE environment {env_id!r}") from exc


def run_side_by_side_rollout(
    *,
    run_id: str,
    current_spec: ModelWatchSpec,
    baseline_spec: ModelWatchSpec,
    current_policy: ActionScoringPolicy,
    baseline_policy: ActionScoringPolicy,
    current_env: NetHackEnv,
    baseline_env: NetHackEnv,
    action_manifest: ActionManifest,
    out_dir: str | Path,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run paired rollouts and write JSONL events, report, and static viewer."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    events_path = target / "events.jsonl"
    latest_path = target / "latest.json"
    report_path = target / "report.json"
    viewer_path = target / "index.html"
    valid_action_ids = action_manifest.valid_action_ids()
    _validate_env_action_space(
        action_manifest=action_manifest,
        env=current_env,
        env_label="current_env",
    )
    _validate_env_action_space(
        action_manifest=action_manifest,
        env=baseline_env,
        env_label="baseline_env",
    )

    current_seeded = seed_nle_env(current_env, seed=seed)
    baseline_seeded = seed_nle_env(baseline_env, seed=seed)
    current_obs = _reset_env(current_env, seed=seed)
    baseline_obs = _reset_env(baseline_env, seed=seed)
    current_reset_obs = current_obs
    baseline_reset_obs = baseline_obs
    current_initial_frame = render_observation_text(current_reset_obs)
    baseline_initial_frame = render_observation_text(baseline_reset_obs)
    paired_initial_state_equal = current_initial_frame == baseline_initial_frame
    current_done = False
    baseline_done = False
    current_cumulative_reward = 0.0
    baseline_cumulative_reward = 0.0
    current_feedback_history: list[dict[str, Any]] = []
    baseline_feedback_history: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    with events_path.open("w", encoding="utf-8") as handle:
        for step_index in range(max_steps):
            current_result = _advance_side(
                spec=current_spec,
                policy=current_policy,
                env=current_env,
                observation=current_obs,
                valid_action_ids=valid_action_ids,
                cumulative_reward=current_cumulative_reward,
                done=current_done,
                feedback_history=current_feedback_history,
            )
            baseline_result = _advance_side(
                spec=baseline_spec,
                policy=baseline_policy,
                env=baseline_env,
                observation=baseline_obs,
                valid_action_ids=valid_action_ids,
                cumulative_reward=baseline_cumulative_reward,
                done=baseline_done,
                feedback_history=baseline_feedback_history,
            )
            _append_policy_feedback(current_feedback_history, current_result)
            _append_policy_feedback(baseline_feedback_history, baseline_result)
            current_obs = current_result.pop("_next_observation")
            baseline_obs = baseline_result.pop("_next_observation")
            current_cumulative_reward = float(current_result["cumulative_reward"])
            baseline_cumulative_reward = float(baseline_result["cumulative_reward"])
            current_done = bool(current_result["done"])
            baseline_done = bool(baseline_result["done"])
            event = {
                "schema_version": COMPARE_EVENT_SCHEMA_VERSION,
                "run_id": run_id,
                "step": step_index,
                "seed": seed,
                "current": current_result,
                "baseline": baseline_result,
            }
            events.append(event)
            line = json.dumps(event, sort_keys=True)
            handle.write(line + "\n")
            handle.flush()
            latest_path.write_text(line + "\n", encoding="utf-8")
            if current_done and baseline_done:
                break

    report = {
        "schema_version": COMPARE_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "seed": seed,
        "max_steps": max_steps,
        "event_count": len(events),
        "current": asdict(current_spec),
        "baseline": asdict(baseline_spec),
        "action_manifest": {
            "env_id": action_manifest.env_id,
            "valid_action_ids": valid_action_ids,
        },
        "deterministic_nle_seed": current_seeded and baseline_seeded,
        "paired_initial_state_equal": paired_initial_state_equal,
        "events_path": str(events_path),
        "latest_path": str(latest_path),
        "viewer_path": str(viewer_path),
        "rollout_metrics": {
            "objective": (
                "maximize live_rollout_utility_v7: score/reward/depth progress "
                "and clean live-progress events while minimizing dirty progress, "
                "HP damage, "
                "death, starvation/fainting, bad messages, wall/no-progress "
                "loops, prompt/menu loops, action collapse, and zero-progress "
                "episodes. Visible-map novelty is logged but carries no scalar "
                "fitness bonus."
            ),
            "current": summarize_rollout_events(
                events,
                side="current",
                reset_observation=current_reset_obs,
            ),
            "baseline": summarize_rollout_events(
                events,
                side="baseline",
                reset_observation=baseline_reset_obs,
            ),
        },
        "status": "completed",
    }
    report["rollout_metrics"]["deltas"] = _rollout_metric_deltas(
        report["rollout_metrics"]["current"],
        report["rollout_metrics"]["baseline"],
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    viewer_path.write_text(
        _render_static_viewer(report=report, events=events), encoding="utf-8"
    )
    return report


def run_side_by_side_rollout_sweep(
    *,
    run_id: str,
    current_spec: ModelWatchSpec,
    baseline_spec: ModelWatchSpec,
    current_policy: ActionScoringPolicy,
    baseline_policy: ActionScoringPolicy,
    make_current_env: Callable[[], NetHackEnv],
    make_baseline_env: Callable[[], NetHackEnv],
    action_manifest: ActionManifest,
    out_dir: str | Path,
    seeds: Sequence[int],
    max_steps: int,
) -> dict[str, Any]:
    """Run paired rollouts across seeds and write an aggregate report."""
    seed_values = parse_seed_list(seeds)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    seed_reports: list[dict[str, Any]] = []
    for seed in seed_values:
        seed_dir = target / f"seed-{seed}"
        current_env = make_current_env()
        baseline_env = make_baseline_env()
        try:
            seed_report = run_side_by_side_rollout(
                run_id=f"{run_id}-seed-{seed}",
                current_spec=current_spec,
                baseline_spec=baseline_spec,
                current_policy=current_policy,
                baseline_policy=baseline_policy,
                current_env=current_env,
                baseline_env=baseline_env,
                action_manifest=action_manifest,
                out_dir=seed_dir,
                seed=seed,
                max_steps=max_steps,
            )
        finally:
            current_env.close()
            baseline_env.close()
        seed_reports.append(_sweep_seed_report_summary(seed_report, seed_dir=seed_dir))

    report = {
        "schema_version": COMPARE_SWEEP_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "seeds": seed_values,
        "seed_count": len(seed_values),
        "max_steps": max_steps,
        "current": asdict(current_spec),
        "baseline": asdict(baseline_spec),
        "action_manifest": {
            "env_id": action_manifest.env_id,
            "valid_action_ids": action_manifest.valid_action_ids(),
        },
        "deterministic_nle_seed_count": sum(
            1 for item in seed_reports if item["deterministic_nle_seed"]
        ),
        "paired_initial_state_equal_count": sum(
            1 for item in seed_reports if item["paired_initial_state_equal"]
        ),
        "total_event_count": sum(int(item["event_count"]) for item in seed_reports),
        "seed_reports": seed_reports,
        "rollout_metrics": _aggregate_sweep_rollout_metrics(seed_reports),
        "report_path": str(target / "sweep_report.json"),
        "status": "completed",
    }
    Path(report["report_path"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def summarize_rollout_events(
    events: Sequence[Mapping[str, Any]],
    *,
    side: str,
    reset_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize reward and damage from one side of a watch rollout."""
    score_initial = _blstat(reset_observation or {}, 9)
    hp_initial = _blstat(reset_observation or {}, 10)
    depth_initial = _blstat(reset_observation or {}, 12)
    side_events = [dict(event.get(side) or {}) for event in events]
    score_values = [_int_or_none(event.get("score")) for event in side_events]
    score_values = [value for value in score_values if value is not None]
    hp_values = [_int_or_none(event.get("hp")) for event in side_events]
    hp_values = [value for value in hp_values if value is not None]
    depth_values = [_int_or_none(event.get("depth")) for event in side_events]
    depth_values = [value for value in depth_values if value is not None]
    last = side_events[-1] if side_events else {}
    score_final = score_values[-1] if score_values else None
    if score_initial is None and score_values:
        score_initial = score_values[0]
    score_delta = (
        int(score_final) - int(score_initial)
        if score_initial is not None and score_final is not None
        else None
    )
    hp_final = hp_values[-1] if hp_values else None
    hp_min = min(hp_values) if hp_values else None
    hp_damage = (
        max(0, int(hp_initial) - int(hp_min))
        if hp_initial is not None and hp_min is not None
        else None
    )
    if depth_initial is None and depth_values:
        depth_initial = depth_values[0]
    depth_max = max(depth_values) if depth_values else None
    depth_delta = (
        max(0, int(depth_max) - int(depth_initial))
        if depth_initial is not None and depth_max is not None
        else 0
    )
    action_counts = Counter(
        str(event["action_id"])
        for event in side_events
        if event.get("action_id") is not None
    )
    step_count = sum(1 for event in side_events if event.get("action_id") is not None)
    message_counts = Counter(str(event.get("message") or "") for event in side_events)
    wall_message_count = sum(1 for event in side_events if _is_wall_event(event))
    bad_message_count = sum(1 for event in side_events if _is_bad_message_event(event))
    non_advancing_steps = sum(
        1
        for event in side_events
        if event.get("action_id") is not None
        and event.get("game_time_advanced") is False
    )
    hunger_warning_count = sum(1 for event in side_events if _is_hunger_warning(event))
    starvation_or_faint_count = sum(
        1 for event in side_events if _is_starvation_or_faint_event(event)
    )
    menu_or_prompt_step_count = sum(
        1 for event in side_events if _is_menu_or_prompt_event(event)
    )
    stuck_menu_or_prompt_loop_count = _count_stuck_menu_or_prompt_loops(side_events)
    visible_map_novelty_count = _visible_map_novelty_count(
        reset_observation=reset_observation,
        side_events=side_events,
    )
    meaningful_event_count = _meaningful_event_count(
        side_events=side_events,
        reset_observation=reset_observation,
    )
    live_progress_event_counts = _live_progress_event_counts(
        side_events=side_events,
        reset_observation=reset_observation,
    )
    raw_live_progress_event_count = live_progress_event_counts["raw"]
    live_progress_event_count = live_progress_event_counts["clean"]
    dirty_live_progress_event_count = live_progress_event_counts["dirty"]
    action_mode_count = max(action_counts.values(), default=0)
    action_repeat_rate = action_mode_count / step_count if step_count else 0.0
    action_collapse_threshold = max(1, int(step_count * 0.75)) if step_count else 0
    action_collapse_excess = max(0, action_mode_count - action_collapse_threshold)
    action_collapse_rate_excess = max(
        0.0, action_repeat_rate - ACTION_COLLAPSE_RATE_THRESHOLD
    )
    wall_message_rate = wall_message_count / step_count if step_count else 0.0
    bad_message_rate = bad_message_count / step_count if step_count else 0.0
    non_advancing_step_rate = non_advancing_steps / step_count if step_count else 0.0
    menu_or_prompt_step_rate = (
        menu_or_prompt_step_count / step_count if step_count else 0.0
    )
    score_delta_for_fitness = (
        float(score_delta)
        if score_delta is not None
        else float(last.get("cumulative_reward", 0.0) or 0.0)
    )
    cumulative_reward = float(last.get("cumulative_reward", 0.0) or 0.0)
    zero_progress_episode = int(
        step_count > 0
        and cumulative_reward <= 0.0
        and score_delta_for_fitness <= 0.0
        and depth_delta <= 0
        and raw_live_progress_event_count == 0
    )
    fitness_components = _rollout_fitness_components(
        score_delta=score_delta_for_fitness,
        cumulative_reward=cumulative_reward,
        depth_delta=depth_delta,
        visible_map_novelty_count=visible_map_novelty_count,
        meaningful_event_count=meaningful_event_count,
        live_progress_event_count=live_progress_event_count,
        hp_damage_observed=hp_damage,
        wall_message_count=wall_message_count,
        wall_message_rate=wall_message_rate,
        bad_message_count=bad_message_count,
        bad_message_rate=bad_message_rate,
        non_advancing_step_count=non_advancing_steps,
        non_advancing_step_rate=non_advancing_step_rate,
        action_collapse_excess=action_collapse_excess,
        action_collapse_rate_excess=action_collapse_rate_excess,
        starvation_or_faint_count=starvation_or_faint_count,
        menu_or_prompt_step_rate=menu_or_prompt_step_rate,
        stuck_menu_or_prompt_loop_count=stuck_menu_or_prompt_loop_count,
        dirty_live_progress_event_count=dirty_live_progress_event_count,
        zero_progress_episode=zero_progress_episode,
        death=last.get("death"),
    )
    return {
        "fitness_objective_version": FITNESS_OBJECTIVE_VERSION,
        "fitness_weights": dict(FITNESS_WEIGHTS),
        "event_count": len(side_events),
        "step_count": step_count,
        "cumulative_reward": cumulative_reward,
        "score_initial": score_initial,
        "score_final": score_final,
        "score_delta": score_delta,
        "score_delta_for_fitness": score_delta_for_fitness,
        "done": bool(last.get("done", False)),
        "death": last.get("death"),
        "hp_initial": hp_initial,
        "hp_final": hp_final,
        "hp_min": hp_min,
        "hp_damage_observed": hp_damage,
        "depth_initial": depth_initial,
        "depth_max": depth_max,
        "depth_delta": depth_delta,
        "action_histogram": dict(sorted(action_counts.items())),
        "action_mode_count": action_mode_count,
        "action_repeat_rate": action_repeat_rate,
        "action_collapse_excess": action_collapse_excess,
        "action_collapse_rate_excess": action_collapse_rate_excess,
        "message_histogram": dict(sorted(message_counts.items())),
        "wall_message_count": wall_message_count,
        "wall_message_rate": wall_message_rate,
        "bad_message_count": bad_message_count,
        "bad_message_rate": bad_message_rate,
        "non_advancing_step_count": non_advancing_steps,
        "non_advancing_step_rate": non_advancing_step_rate,
        "hunger_warning_count": hunger_warning_count,
        "starvation_or_faint_count": starvation_or_faint_count,
        "menu_or_prompt_step_count": menu_or_prompt_step_count,
        "menu_or_prompt_step_rate": menu_or_prompt_step_rate,
        "stuck_menu_or_prompt_loop_count": stuck_menu_or_prompt_loop_count,
        "visible_map_novelty_count": visible_map_novelty_count,
        "visible_map_novelty_bonus_count": min(
            visible_map_novelty_count, MAX_VISIBLE_MAP_NOVELTY_BONUS_COUNT
        ),
        "explored_tile_delta_proxy": visible_map_novelty_count,
        "meaningful_event_count": meaningful_event_count,
        "meaningful_event_bonus_count": min(
            meaningful_event_count, MAX_MEANINGFUL_EVENT_BONUS_COUNT
        ),
        "raw_live_progress_event_count": raw_live_progress_event_count,
        "clean_live_progress_event_count": live_progress_event_count,
        "live_progress_event_count": live_progress_event_count,
        "dirty_live_progress_event_count": dirty_live_progress_event_count,
        "zero_progress_episode": zero_progress_episode,
        "fitness_components": fitness_components,
        "fitness_score": sum(fitness_components.values()),
    }


def _rollout_fitness_components(
    *,
    score_delta: float,
    cumulative_reward: float,
    depth_delta: int,
    visible_map_novelty_count: int,
    meaningful_event_count: int,
    live_progress_event_count: int,
    hp_damage_observed: int | None,
    wall_message_count: int,
    wall_message_rate: float,
    bad_message_count: int,
    bad_message_rate: float,
    non_advancing_step_count: int,
    non_advancing_step_rate: float,
    action_collapse_excess: int,
    action_collapse_rate_excess: float,
    starvation_or_faint_count: int,
    menu_or_prompt_step_rate: float,
    stuck_menu_or_prompt_loop_count: int,
    dirty_live_progress_event_count: int,
    zero_progress_episode: int,
    death: Any,
) -> dict[str, float]:
    """Score live rollouts with dense penalties for known bad NetHack loops."""
    visible_map_novelty_bonus_count = min(
        visible_map_novelty_count, MAX_VISIBLE_MAP_NOVELTY_BONUS_COUNT
    )
    meaningful_event_bonus_count = min(
        meaningful_event_count, MAX_MEANINGFUL_EVENT_BONUS_COUNT
    )
    return {
        "normalized_score_delta_bonus": FITNESS_WEIGHTS["normalized_score_delta"]
        * _signed_log1p(score_delta),
        "cumulative_reward_bonus": FITNESS_WEIGHTS["cumulative_reward"]
        * _signed_log1p(cumulative_reward),
        "depth_delta_bonus": FITNESS_WEIGHTS["depth_delta"] * float(depth_delta),
        "visible_map_novelty_bonus": FITNESS_WEIGHTS["visible_map_novelty"]
        * float(visible_map_novelty_bonus_count),
        "meaningful_event_bonus": FITNESS_WEIGHTS["meaningful_event"]
        * float(meaningful_event_bonus_count),
        "live_progress_event_bonus": FITNESS_WEIGHTS["live_progress_event"]
        * float(live_progress_event_count),
        "hp_damage_penalty": FITNESS_WEIGHTS["hp_damage"]
        * float(hp_damage_observed or 0),
        "wall_message_penalty": FITNESS_WEIGHTS["wall_or_solid_stone_message"]
        * float(wall_message_count),
        "wall_message_rate_penalty": FITNESS_WEIGHTS["wall_message_rate"]
        * float(wall_message_rate),
        "bad_message_penalty": FITNESS_WEIGHTS["bad_message"]
        * float(bad_message_count),
        "bad_message_rate_penalty": FITNESS_WEIGHTS["bad_message_rate"]
        * float(bad_message_rate),
        "non_advancing_step_penalty": FITNESS_WEIGHTS["non_advancing_step"]
        * float(non_advancing_step_count),
        "non_advancing_step_rate_penalty": FITNESS_WEIGHTS["non_advancing_step_rate"]
        * float(non_advancing_step_rate),
        "action_collapse_penalty": FITNESS_WEIGHTS["action_collapse_excess"]
        * float(action_collapse_excess),
        "action_collapse_rate_penalty": FITNESS_WEIGHTS["action_collapse_rate_excess"]
        * float(action_collapse_rate_excess),
        "starvation_or_faint_penalty": FITNESS_WEIGHTS["starvation_or_faint"]
        * float(starvation_or_faint_count),
        "menu_or_prompt_step_rate_penalty": FITNESS_WEIGHTS["menu_or_prompt_step_rate"]
        * float(menu_or_prompt_step_rate),
        "stuck_menu_or_prompt_loop_penalty": FITNESS_WEIGHTS[
            "stuck_menu_or_prompt_loop"
        ]
        * float(stuck_menu_or_prompt_loop_count),
        "dirty_live_progress_event_penalty": FITNESS_WEIGHTS[
            "dirty_live_progress_event"
        ]
        * float(dirty_live_progress_event_count),
        "zero_progress_episode_penalty": FITNESS_WEIGHTS["zero_progress_episode"]
        * float(zero_progress_episode),
        "death_penalty": FITNESS_WEIGHTS["death"] if death else 0.0,
    }


def _rollout_metric_deltas(
    current: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    reward_delta = float(current.get("cumulative_reward", 0.0) or 0.0) - float(
        baseline.get("cumulative_reward", 0.0) or 0.0
    )
    damage_delta = _delta_if_numeric(
        current.get("hp_damage_observed"),
        baseline.get("hp_damage_observed"),
    )
    return {
        "cumulative_reward": reward_delta,
        "score_delta": _delta_if_numeric(
            current.get("score_delta"),
            baseline.get("score_delta"),
        ),
        "hp_damage_observed": damage_delta,
        "fitness_score": _delta_if_numeric(
            current.get("fitness_score"),
            baseline.get("fitness_score"),
        ),
        "depth_delta": _delta_if_numeric(
            current.get("depth_delta"),
            baseline.get("depth_delta"),
        ),
        "depth_max": _delta_if_numeric(
            current.get("depth_max"),
            baseline.get("depth_max"),
        ),
        "wall_message_count": _delta_if_numeric(
            current.get("wall_message_count"),
            baseline.get("wall_message_count"),
        ),
        "wall_message_rate": _delta_if_numeric(
            current.get("wall_message_rate"),
            baseline.get("wall_message_rate"),
        ),
        "bad_message_count": _delta_if_numeric(
            current.get("bad_message_count"),
            baseline.get("bad_message_count"),
        ),
        "bad_message_rate": _delta_if_numeric(
            current.get("bad_message_rate"),
            baseline.get("bad_message_rate"),
        ),
        "non_advancing_step_count": _delta_if_numeric(
            current.get("non_advancing_step_count"),
            baseline.get("non_advancing_step_count"),
        ),
        "non_advancing_step_rate": _delta_if_numeric(
            current.get("non_advancing_step_rate"),
            baseline.get("non_advancing_step_rate"),
        ),
        "action_repeat_rate": _delta_if_numeric(
            current.get("action_repeat_rate"),
            baseline.get("action_repeat_rate"),
        ),
        "action_collapse_excess": _delta_if_numeric(
            current.get("action_collapse_excess"),
            baseline.get("action_collapse_excess"),
        ),
        "action_collapse_rate_excess": _delta_if_numeric(
            current.get("action_collapse_rate_excess"),
            baseline.get("action_collapse_rate_excess"),
        ),
        "visible_map_novelty_count": _delta_if_numeric(
            current.get("visible_map_novelty_count"),
            baseline.get("visible_map_novelty_count"),
        ),
        "meaningful_event_count": _delta_if_numeric(
            current.get("meaningful_event_count"),
            baseline.get("meaningful_event_count"),
        ),
        "raw_live_progress_event_count": _delta_if_numeric(
            current.get("raw_live_progress_event_count"),
            baseline.get("raw_live_progress_event_count"),
        ),
        "clean_live_progress_event_count": _delta_if_numeric(
            current.get("clean_live_progress_event_count"),
            baseline.get("clean_live_progress_event_count"),
        ),
        "starvation_or_faint_count": _delta_if_numeric(
            current.get("starvation_or_faint_count"),
            baseline.get("starvation_or_faint_count"),
        ),
        "menu_or_prompt_step_count": _delta_if_numeric(
            current.get("menu_or_prompt_step_count"),
            baseline.get("menu_or_prompt_step_count"),
        ),
        "menu_or_prompt_step_rate": _delta_if_numeric(
            current.get("menu_or_prompt_step_rate"),
            baseline.get("menu_or_prompt_step_rate"),
        ),
        "stuck_menu_or_prompt_loop_count": _delta_if_numeric(
            current.get("stuck_menu_or_prompt_loop_count"),
            baseline.get("stuck_menu_or_prompt_loop_count"),
        ),
        "live_progress_event_count": _delta_if_numeric(
            current.get("live_progress_event_count"),
            baseline.get("live_progress_event_count"),
        ),
        "dirty_live_progress_event_count": _delta_if_numeric(
            current.get("dirty_live_progress_event_count"),
            baseline.get("dirty_live_progress_event_count"),
        ),
        "zero_progress_episode": _delta_if_numeric(
            current.get("zero_progress_episode"),
            baseline.get("zero_progress_episode"),
        ),
    }


def _sweep_seed_report_summary(
    report: Mapping[str, Any],
    *,
    seed_dir: Path,
) -> dict[str, Any]:
    return {
        "seed": int(report["seed"]),
        "run_id": str(report["run_id"]),
        "event_count": int(report.get("event_count", 0) or 0),
        "deterministic_nle_seed": bool(report.get("deterministic_nle_seed", False)),
        "paired_initial_state_equal": bool(
            report.get("paired_initial_state_equal", False)
        ),
        "events_path": str(seed_dir / "events.jsonl"),
        "latest_path": str(seed_dir / "latest.json"),
        "viewer_path": str(seed_dir / "index.html"),
        "report_path": str(seed_dir / "report.json"),
        "rollout_metrics": dict(report.get("rollout_metrics") or {}),
    }


def _aggregate_sweep_rollout_metrics(
    seed_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rollout_metrics = [
        dict(report.get("rollout_metrics") or {}) for report in seed_reports
    ]
    current_metrics = [
        dict(metrics.get("current") or {}) for metrics in rollout_metrics
    ]
    baseline_metrics = [
        dict(metrics.get("baseline") or {}) for metrics in rollout_metrics
    ]
    delta_metrics = [dict(metrics.get("deltas") or {}) for metrics in rollout_metrics]
    return {
        "aggregation": "mean_over_seed_reports",
        "seed_count": len(seed_reports),
        "current": _mean_numeric_metrics(current_metrics),
        "baseline": _mean_numeric_metrics(baseline_metrics),
        "deltas": _mean_numeric_metrics(delta_metrics),
    }


def _mean_numeric_metrics(
    metrics_by_seed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values_by_key: dict[str, list[float]] = {}
    for metrics in metrics_by_seed:
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                values_by_key.setdefault(key, []).append(float(value))
    means = {
        key: sum(values) / len(values)
        for key, values in sorted(values_by_key.items())
        if values
    }
    versions = {
        str(metrics.get("fitness_objective_version"))
        for metrics in metrics_by_seed
        if metrics.get("fitness_objective_version")
    }
    if len(versions) == 1:
        means["fitness_objective_version"] = next(iter(versions))
    for metrics in metrics_by_seed:
        weights = metrics.get("fitness_weights")
        if isinstance(weights, Mapping):
            means["fitness_weights"] = dict(weights)
            break
    return means


def _delta_if_numeric(value: Any, baseline: Any) -> float:
    if value is None or baseline is None:
        return 0.0
    return float(value) - float(baseline)


def seed_nle_env(env: NetHackEnv, *, seed: int) -> bool:
    """Set NLE core, display, and level-generation seeds when the env allows it."""
    target = getattr(env, "unwrapped", env)
    seed_fn = getattr(target, "seed", None)
    if seed_fn is None:
        return False
    try:
        seed_fn(core=seed, disp=seed, reseed=False, lgen=seed)
    except (RuntimeError, TypeError):
        return False
    return True


def write_compare_watch_contract(
    *,
    run_id: str,
    current_checkpoint: str | None,
    action_manifest_path: str | Path,
    out_dir: str | Path,
    model_name: str = "google/gemma-4-E4b-it",
    env_id: str = "NetHackChallenge-v0",
    character: str = DEFAULT_NLE_CHARACTER,
    seed: int = 20260615,
    max_steps: int = 80,
) -> dict[str, Any]:
    """Write a dry-run contract for a compare-watch run without heavy deps."""
    manifest = load_action_manifest(action_manifest_path)
    validate_action_manifest_env_id(manifest, env_id=env_id)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": COMPARE_CONTRACT_SCHEMA_VERSION,
        "run_id": run_id,
        "env_id": env_id,
        "character": character,
        "seed": seed,
        "max_steps": max_steps,
        "current": asdict(
            ModelWatchSpec(
                role="current",
                model_name=model_name,
                adapter_checkpoint=current_checkpoint,
            )
        ),
        "baseline": asdict(
            ModelWatchSpec(
                role="baseline",
                model_name=model_name,
                adapter_checkpoint=None,
            )
        ),
        "action_manifest_path": str(action_manifest_path),
        "valid_action_ids": manifest.valid_action_ids(),
        "status": "requires_nle_env_and_model_deps_to_run",
    }
    (target / "compare_watch_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def write_compare_watch_sweep_contract(
    *,
    run_id: str,
    current_checkpoint: str | None,
    action_manifest_path: str | Path,
    out_dir: str | Path,
    seeds: str | Sequence[int],
    model_name: str = "google/gemma-4-E4b-it",
    env_id: str = "NetHackChallenge-v0",
    character: str = DEFAULT_NLE_CHARACTER,
    max_steps: int = 80,
) -> dict[str, Any]:
    """Write a dry-run multi-seed watch contract without heavy deps."""
    manifest = load_action_manifest(action_manifest_path)
    validate_action_manifest_env_id(manifest, env_id=env_id)
    seed_values = parse_seed_list(seeds)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": COMPARE_SWEEP_CONTRACT_SCHEMA_VERSION,
        "run_id": run_id,
        "env_id": env_id,
        "character": character,
        "seeds": seed_values,
        "max_steps": max_steps,
        "current": asdict(
            ModelWatchSpec(
                role="current",
                model_name=model_name,
                adapter_checkpoint=current_checkpoint,
            )
        ),
        "baseline": asdict(
            ModelWatchSpec(
                role="baseline",
                model_name=model_name,
                adapter_checkpoint=None,
            )
        ),
        "action_manifest_path": str(action_manifest_path),
        "valid_action_ids": manifest.valid_action_ids(),
        "report_path": str(target / "sweep_report.json"),
        "seed_report_dirs": [str(target / f"seed-{seed}") for seed in seed_values],
        "status": "requires_nle_env_and_model_deps_to_run",
    }
    (target / "compare_watch_sweep_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def _advance_side(
    *,
    spec: ModelWatchSpec,
    policy: ActionScoringPolicy,
    env: NetHackEnv,
    observation: dict[str, Any],
    valid_action_ids: list[int],
    cumulative_reward: float,
    done: bool,
    feedback_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if done:
        terminal_frame = render_observation_text(observation)
        return {
            **asdict(spec),
            "prompt_terminal_frame": terminal_frame,
            "policy_observation_text": terminal_frame,
            "prompt_score": _blstat(observation, 9),
            "prompt_hp": _blstat(observation, 10),
            "prompt_depth": _blstat(observation, 12),
            "terminal_frame": terminal_frame,
            "action_id": None,
            "action_label": None,
            "reward": 0.0,
            "cumulative_reward": cumulative_reward,
            "done": True,
            "death": None,
            "score": _blstat(observation, 9),
            "hp": _blstat(observation, 10),
            "depth": _blstat(observation, 12),
            "message": _message_from_observation(observation),
            "hunger": None,
            "menu_open": None,
            "game_time_advanced": False,
            "policy_feedback_length": len(feedback_history),
            "policy_feedback": list(feedback_history),
            "_next_observation": observation,
        }

    observation_text = render_observation_text(observation)
    policy_observation_text = build_policy_observation_with_feedback(
        observation_text=observation_text,
        feedback_history=feedback_history,
    )
    scores = policy.score_actions(
        observation_text=policy_observation_text,
        valid_action_ids=valid_action_ids,
    )
    action_id = select_action_id(
        scores_by_action_id=scores,
        valid_action_ids=valid_action_ids,
    )
    next_observation, reward, step_done, info = _step_env(env, action_id)
    next_cumulative_reward = cumulative_reward + reward
    return {
        **asdict(spec),
        "prompt_terminal_frame": observation_text,
        "policy_observation_text": policy_observation_text,
        "prompt_score": _blstat(observation, 9),
        "prompt_hp": _blstat(observation, 10),
        "prompt_depth": _blstat(observation, 12),
        "terminal_frame": render_observation_text(next_observation),
        "action_id": action_id,
        "action_label": format_action_candidate(action_id),
        "reward": reward,
        "cumulative_reward": next_cumulative_reward,
        "done": step_done,
        "death": _info_value(info, "death"),
        "score": _info_value(info, "score", _blstat(next_observation, 9)),
        "hp": _info_value(info, "hp", _blstat(next_observation, 10)),
        "depth": _info_value(info, "depth", _blstat(next_observation, 12)),
        "message": _info_value(
            info, "message", _message_from_observation(next_observation)
        ),
        "hunger": _info_value(info, "hunger"),
        "menu_open": _info_value(info, "menu_open"),
        "game_time_advanced": bool(_info_value(info, "game_time_advanced", True)),
        "policy_feedback_length": len(feedback_history),
        "policy_feedback": list(feedback_history),
        "_next_observation": next_observation,
    }


def _append_policy_feedback(
    feedback_history: list[dict[str, Any]],
    result: Mapping[str, Any],
) -> None:
    action_id = result.get("action_id")
    if action_id is None:
        return
    feedback_history.append(
        {
            "action_id": int(action_id),
            "reward": float(result.get("reward", 0.0) or 0.0),
            "cumulative_reward": float(result.get("cumulative_reward", 0.0) or 0.0),
            "message": str(result.get("message") or ""),
            "hp": result.get("hp"),
            "depth": result.get("depth"),
            "game_time_advanced": bool(result.get("game_time_advanced", False)),
        }
    )
    del feedback_history[:-MAX_POLICY_FEEDBACK_ITEMS]


def _reset_env(env: NetHackEnv, *, seed: int) -> dict[str, Any]:
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        return dict(reset_result[0])
    return dict(reset_result)


def _step_env(
    env: NetHackEnv, action_id: int
) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    step_result = env.step(action_id)
    if not isinstance(step_result, tuple):
        raise RuntimeError("NLE env step must return a tuple")
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        done = bool(terminated) or bool(truncated)
    elif len(step_result) == 4:
        observation, reward, done, info = step_result
    else:
        raise RuntimeError(f"NLE env step returned {len(step_result)} values")
    return dict(observation), float(reward), bool(done), dict(info or {})


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        + "\nassistant: "
    )


def _pad_token_id(tokenizer: Any) -> int:
    for attribute_name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, attribute_name, None)
        if value is not None:
            return int(value)
    return 0


def _release_torch_cuda_cache(torch: Any) -> None:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    try:
        available = bool(is_available()) if callable(is_available) else True
    except RuntimeError:
        available = False
    if not available:
        return
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _blstat(observation: Mapping[str, Any], index: int) -> int | None:
    values = observation.get("blstats")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    if len(values) <= index:
        return None
    try:
        return int(values[index])
    except (TypeError, ValueError):
        return None


def _message_from_observation(observation: Mapping[str, Any]) -> str:
    values = observation.get("message")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not values:
        return ""
    chars: list[str] = []
    for value in values:
        integer = int(value)
        if integer:
            chars.append(chr(integer))
    return "".join(chars).strip()


def _info_value(info: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = info.get(key, default)
    if hasattr(value, "item"):
        return value.item()
    return value


def _validate_env_action_space(
    *,
    action_manifest: ActionManifest,
    env: NetHackEnv,
    env_label: str,
) -> None:
    action_count = _env_action_count(env)
    if action_count is None:
        return
    valid_action_ids = action_manifest.valid_action_ids()
    if not valid_action_ids:
        return
    max_action_id = max(valid_action_ids)
    if max_action_id >= action_count:
        raise ValueError(
            f"{env_label} action space has {action_count} actions, but action "
            f"manifest {action_manifest.env_id!r} contains action_id {max_action_id}; "
            "use a matching NLE env_id/action_manifest pair"
        )


def _env_action_count(env: NetHackEnv) -> int | None:
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        actions = getattr(target, "actions", None)
        if actions is None:
            continue
        try:
            action_count = len(actions)
        except TypeError:
            continue
        if action_count > 0:
            return int(action_count)
    return None


def _signed_log1p(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _is_wall_message(message: Any) -> bool:
    text = str(message or "").lower()
    return "wall" in text or "solid stone" in text


def _is_wall_event(event: Mapping[str, Any]) -> bool:
    return _is_wall_message(event.get("message")) or _is_wall_message(
        event.get("terminal_frame")
    )


def _is_bad_message_event(event: Mapping[str, Any]) -> bool:
    if _is_wall_event(event) or _is_menu_or_prompt_event(event):
        return True
    text = _event_status_text(event)
    markers = (
        "you don't have",
        "you cannot",
        "you can't",
        "can't ",
        "cannot ",
        "never mind",
        "what a strange direction",
        "not possible",
        "nothing happens",
        "there is nothing",
        "don't know how",
        "not enough",
    )
    return any(marker in text for marker in markers)


def _is_hunger_warning(event: Mapping[str, Any]) -> bool:
    text = _event_status_text(event)
    if "not hungry" in text:
        return False
    return "hungry" in text or "hunger" in text or "weak from severe hunger" in text


def _is_starvation_or_faint_event(event: Mapping[str, Any]) -> bool:
    text = _event_status_text(event)
    return "starv" in text or "faint" in text


def _is_menu_or_prompt_event(event: Mapping[str, Any]) -> bool:
    if event.get("menu_open") is True:
        return True
    text = _event_status_text(event)
    markers = (
        "--more--",
        "-- more --",
        "extended commands list",
        "extended commands",
        "voluntary challenges:",
        "(1 of ",
        "(2 of ",
        "(3 of ",
        "(4 of ",
        "(5 of ",
        "what do you want",
        "in what direction",
        "which object",
        "pick up",
        "really ",
        "call ",
        "name ",
        "[yn",
        "[ynq",
    )
    return any(marker in text for marker in markers)


def _event_status_text(event: Mapping[str, Any]) -> str:
    values = (
        event.get("message"),
        event.get("hunger"),
        event.get("death"),
        event.get("terminal_frame"),
    )
    return " ".join(str(value or "") for value in values).lower()


def _count_stuck_menu_or_prompt_loops(events: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    previous_signature: tuple[Any, str] | None = None
    for event in events:
        if not _is_menu_or_prompt_event(event):
            previous_signature = None
            continue
        signature = (event.get("action_id"), str(event.get("message") or "").lower())
        if previous_signature == signature:
            count += 1
        previous_signature = signature
    return count


def _visible_map_novelty_count(
    *,
    reset_observation: Mapping[str, Any] | None,
    side_events: Sequence[Mapping[str, Any]],
) -> int:
    seen = (
        _visible_map_tile_keys(render_observation_text(reset_observation))
        if reset_observation
        else set()
    )
    initial_count = len(seen)
    for event in side_events:
        frame = event.get("terminal_frame")
        if isinstance(frame, str):
            seen.update(_visible_map_tile_keys(frame))
    return max(0, len(seen) - initial_count)


def _meaningful_event_count(
    *,
    side_events: Sequence[Mapping[str, Any]],
    reset_observation: Mapping[str, Any] | None,
) -> int:
    count = 0
    previous_score = _blstat(reset_observation or {}, 9)
    previous_depth = _blstat(reset_observation or {}, 12)
    for event in side_events:
        if event.get("action_id") is None:
            continue
        current_score = _int_or_none(event.get("score"))
        current_depth = _int_or_none(event.get("depth"))
        positive_reward = float(event.get("reward", 0.0) or 0.0) > 0.0
        score_increased = (
            previous_score is not None
            and current_score is not None
            and current_score > previous_score
        )
        depth_increased = (
            previous_depth is not None
            and current_depth is not None
            and current_depth > previous_depth
        )
        has_progress = positive_reward or score_increased or depth_increased
        if has_progress and not _is_dirty_progress_event(event):
            count += 1
        if current_score is not None:
            previous_score = current_score
        if current_depth is not None:
            previous_depth = current_depth
    return count


def _live_progress_event_counts(
    *,
    side_events: Sequence[Mapping[str, Any]],
    reset_observation: Mapping[str, Any] | None,
) -> dict[str, int]:
    raw = 0
    clean = 0
    dirty = 0
    previous_score = _blstat(reset_observation or {}, 9)
    previous_depth = _blstat(reset_observation or {}, 12)
    for event in side_events:
        if event.get("action_id") is None:
            continue
        current_score = _int_or_none(event.get("score"))
        current_depth = _int_or_none(event.get("depth"))
        positive_reward = float(event.get("reward", 0.0) or 0.0) > 0.0
        score_increased = (
            previous_score is not None
            and current_score is not None
            and current_score > previous_score
        )
        depth_increased = (
            previous_depth is not None
            and current_depth is not None
            and current_depth > previous_depth
        )
        has_progress = positive_reward or score_increased or depth_increased
        if has_progress:
            raw += 1
            if _is_dirty_progress_event(event):
                dirty += 1
            else:
                clean += 1
        if current_score is not None:
            previous_score = current_score
        if current_depth is not None:
            previous_depth = current_depth
    return {"raw": raw, "clean": clean, "dirty": dirty}


def _is_dirty_progress_event(event: Mapping[str, Any]) -> bool:
    return (
        _is_bad_message_event(event)
        or event.get("game_time_advanced") is False
        or bool(event.get("death"))
    )


def _visible_map_tile_keys(frame: str) -> set[tuple[int, int, str]]:
    lines = frame.splitlines()
    try:
        start = lines.index("MAP:") + 1
    except ValueError:
        return set()
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index] == "MESSAGE:":
            end = index
            break
    keys: set[tuple[int, int, str]] = set()
    for row_index, line in enumerate(lines[start:end]):
        if not _is_likely_visible_map_row(line):
            continue
        for column_index, char in enumerate(line):
            if char in MAP_NOVELTY_CHARS:
                keys.add((row_index, column_index, char))
    return keys


def _is_likely_visible_map_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    map_char_count = sum(1 for char in stripped if char in MAP_NOVELTY_CHARS)
    if map_char_count == 0:
        return False
    if "@" in stripped:
        return True
    if any(char in stripped for char in "|-+<>"):
        return map_char_count >= 2
    return map_char_count >= 3 and map_char_count / len(stripped) >= 0.5


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _render_static_viewer(
    *, report: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> str:
    report_json = json.dumps(report, sort_keys=True)
    events_json = json.dumps(list(events), sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NetHack Checkpoint Compare</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101214;
      --panel: #181d21;
      --line: #34404a;
      --text: #e7ecef;
      --muted: #a7b0b7;
      --accent-current: #67c1a3;
      --accent-baseline: #d6a65f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 16px 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px 24px;
      align-items: baseline;
    }}
    h1 {{
      font-size: 18px;
      margin: 0;
      font-weight: 700;
    }}
    .meta {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
    }}
    main {{
      padding: 16px 20px 24px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(160px, 1fr) auto;
      gap: 16px;
      align-items: center;
      margin-bottom: 16px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    section {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      min-width: 0;
      overflow: hidden;
    }}
    section h2 {{
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    section.current h2 {{ color: var(--accent-current); }}
    section.baseline h2 {{ color: var(--accent-baseline); }}
    dl {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      margin: 0;
      background: var(--line);
    }}
    dt, dd {{
      margin: 0;
      background: var(--panel);
      padding: 8px 10px;
      min-width: 0;
    }}
    dt {{
      color: var(--muted);
    }}
    pre {{
      margin: 0;
      padding: 12px;
      min-height: 340px;
      max-height: 70vh;
      overflow: auto;
      white-space: pre-wrap;
      color: #f4f7f9;
      background: #080a0b;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>NetHack Checkpoint Compare</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <div class="controls">
      <input id="step" type="range" min="0" value="0">
      <strong id="stepLabel"></strong>
    </div>
    <div class="grid">
      <section class="current">
        <h2>Current Checkpoint</h2>
        <dl id="currentStats"></dl>
        <pre id="currentFrame"></pre>
      </section>
      <section class="baseline">
        <h2>Baseline Gemma</h2>
        <dl id="baselineStats"></dl>
        <pre id="baselineFrame"></pre>
      </section>
    </div>
  </main>
  <script type="application/json" id="report-data">{report_json}</script>
  <script type="application/json" id="event-data">{events_json}</script>
  <script>
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const events = JSON.parse(document.getElementById("event-data").textContent);
    const slider = document.getElementById("step");
    const stepLabel = document.getElementById("stepLabel");
    slider.max = Math.max(events.length - 1, 0);
    document.getElementById("meta").textContent =
      `run ${{report.run_id}} | seed ${{report.seed}} | events ${{events.length}}`;

    function statsHtml(side) {{
      const pairs = [
        ["action", side.action_id ?? "done"],
        ["reward", side.reward],
        ["total", side.cumulative_reward],
        ["score", side.score ?? ""],
        ["done", side.done],
        ["hp", side.hp ?? ""],
        ["depth", side.depth ?? ""],
        ["hunger", side.hunger ?? ""],
        ["message", side.message ?? ""],
      ];
      return pairs.map(([key, value]) => `<dt>${{key}}</dt><dd>${{value}}</dd>`).join("");
    }}

    function render(index) {{
      const event = events[index] || {{ step: 0, current: {{}}, baseline: {{}} }};
      stepLabel.textContent = `step ${{event.step ?? 0}}`;
      document.getElementById("currentStats").innerHTML = statsHtml(event.current);
      document.getElementById("baselineStats").innerHTML = statsHtml(event.baseline);
      document.getElementById("currentFrame").textContent = event.current.terminal_frame || "";
      document.getElementById("baselineFrame").textContent = event.baseline.terminal_frame || "";
    }}
    slider.addEventListener("input", () => render(Number(slider.value)));
    render(0);
  </script>
</body>
</html>
"""
