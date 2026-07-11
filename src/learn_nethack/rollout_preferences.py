"""Build preference rows from watchable NetHack rollout comparisons."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from learn_nethack.compare_watch import (
    FITNESS_OBJECTIVE_VERSION,
    FITNESS_WEIGHTS,
    MAP_NOVELTY_CHARS,
    MAX_MEANINGFUL_EVENT_BONUS_COUNT,
    MAX_VISIBLE_MAP_NOVELTY_BONUS_COUNT,
    build_policy_messages,
    format_action_candidate,
)
from learn_nethack.sft_rows import build_policy_prompt


PREFERENCE_ROW_SCHEMA_VERSION = "learn-nethack.policy-action-preference-row.v1"
PREFERENCE_BUILD_REPORT_SCHEMA_VERSION = (
    "learn-nethack.rollout-preference-build-report.v1"
)
MIN_PREFERENCE_UTILITY_MARGIN = 0.05


@dataclass(frozen=True)
class TransitionUtilityBreakdown:
    """Scalar transition utility plus quality signals used for preferences."""

    value: float
    components: dict[str, float]
    positive_signal: bool
    bad_signal: bool


def build_policy_action_preference_rows(
    watch_report: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build same-prompt DPO-style policy preference rows from a watch report."""
    valid_action_ids = _valid_action_ids(watch_report)
    events = _events_from_watch_report(watch_report)
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    run_id = str(watch_report.get("run_id") or "unknown")

    for event in events:
        current = dict(event.get("current") or {})
        baseline = dict(event.get("baseline") or {})
        step = int(event.get("step", len(rows)))
        current_action = _int_or_none(current.get("action_id"))
        baseline_action = _int_or_none(baseline.get("action_id"))
        if current_action is None or baseline_action is None:
            skipped["missing_action"] += 1
            continue
        if current_action == baseline_action:
            skipped["same_action"] += 1
            continue
        current_prompt = _policy_prompt_text(current, valid_action_ids=valid_action_ids)
        baseline_prompt = _policy_prompt_text(
            baseline,
            valid_action_ids=valid_action_ids,
        )
        if current_prompt is None or baseline_prompt is None:
            skipped["missing_prompt"] += 1
            continue
        if current_prompt != baseline_prompt:
            skipped["divergent_prompt"] += 1
            continue

        current_breakdown = transition_utility_breakdown(current)
        baseline_breakdown = transition_utility_breakdown(baseline)
        current_utility = current_breakdown.value
        baseline_utility = baseline_breakdown.value
        utility_margin = abs(current_utility - baseline_utility)
        if utility_margin < MIN_PREFERENCE_UTILITY_MARGIN:
            skipped["utility_tie"] += 1
            continue
        if current_utility > baseline_utility:
            chosen_side = "current"
            rejected_side = "baseline"
            chosen = current
            rejected = baseline
            chosen_breakdown = current_breakdown
            rejected_breakdown = baseline_breakdown
            chosen_utility = current_utility
            rejected_utility = baseline_utility
            chosen_action = current_action
            rejected_action = baseline_action
        else:
            chosen_side = "baseline"
            rejected_side = "current"
            chosen = baseline
            rejected = current
            chosen_breakdown = baseline_breakdown
            rejected_breakdown = current_breakdown
            chosen_utility = baseline_utility
            rejected_utility = current_utility
            chosen_action = baseline_action
            rejected_action = current_action
        if not _is_high_quality_preference(chosen_breakdown, rejected_breakdown):
            skipped["low_quality_preference"] += 1
            continue

        rows.append(
            {
                "schema_version": PREFERENCE_ROW_SCHEMA_VERSION,
                "task": "policy_action_preference",
                "messages": build_policy_messages(user_prompt=current_prompt),
                "chosen": {
                    "role": "assistant",
                    "content": format_action_candidate(chosen_action),
                },
                "rejected": {
                    "role": "assistant",
                    "content": format_action_candidate(rejected_action),
                },
                "metadata": {
                    "run_id": run_id,
                    "step": step,
                    "chosen_side": chosen_side,
                    "rejected_side": rejected_side,
                    "chosen_action_id": chosen_action,
                    "rejected_action_id": rejected_action,
                    "chosen_utility": chosen_utility,
                    "rejected_utility": rejected_utility,
                    "utility_margin": chosen_utility - rejected_utility,
                    "minimum_utility_margin": MIN_PREFERENCE_UTILITY_MARGIN,
                    "chosen_positive_signal": chosen_breakdown.positive_signal,
                    "rejected_positive_signal": rejected_breakdown.positive_signal,
                    "chosen_bad_signal": chosen_breakdown.bad_signal,
                    "rejected_bad_signal": rejected_breakdown.bad_signal,
                    "chosen_utility_components": chosen_breakdown.components,
                    "rejected_utility_components": rejected_breakdown.components,
                    "fitness_objective_version": _fitness_objective_version(
                        watch_report
                    ),
                    "chosen_message": str(chosen.get("message") or ""),
                    "rejected_message": str(rejected.get("message") or ""),
                },
            }
        )

    report = {
        "schema_version": PREFERENCE_BUILD_REPORT_SCHEMA_VERSION,
        "source_run_id": run_id,
        "fitness_objective_version": _fitness_objective_version(watch_report),
        "event_count": len(events),
        "row_count": len(rows),
        "skipped_counts": dict(sorted(skipped.items())),
        "valid_action_ids": valid_action_ids,
        "preference_quality_filter": {
            "minimum_utility_margin": MIN_PREFERENCE_UTILITY_MARGIN,
            "requires": (
                "chosen positive progress signal, or clean avoidance of a worse "
                "failure without chosen-side bad signal"
            ),
        },
    }
    return rows, report


def write_rollout_preference_jsonl(
    *,
    watch_report: Mapping[str, Any],
    out_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Write rollout preference rows and a small build report."""
    rows, report = build_policy_action_preference_rows(watch_report)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_with_paths = {
        **report,
        "out_path": str(target),
        "report_path": str(report_target),
    }
    report_target.write_text(
        json.dumps(report_with_paths, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_with_paths


def transition_utility(event_side: Mapping[str, Any]) -> float:
    """Score one selected action outcome with the live-utility components."""
    return transition_utility_breakdown(event_side).value


def transition_utility_breakdown(
    event_side: Mapping[str, Any],
) -> TransitionUtilityBreakdown:
    """Score one selected action outcome and expose preference-quality signals."""
    score_delta = _delta(event_side, current_key="score", prompt_key="prompt_score")
    depth_delta = max(
        0.0,
        _delta(event_side, current_key="depth", prompt_key="prompt_depth"),
    )
    hp_damage = max(
        0.0,
        -_delta(event_side, current_key="hp", prompt_key="prompt_hp"),
    )
    reward = _float_or_none(event_side.get("reward")) or 0.0
    wall_count = 1.0 if _is_wall_event(event_side) else 0.0
    bad_message_count = 1.0 if _is_bad_message_event(event_side) else 0.0
    non_advancing_count = (
        1.0
        if event_side.get("game_time_advanced") is False
        and event_side.get("action_id") is not None
        else 0.0
    )
    starvation_count = 1.0 if _is_starvation_or_faint_event(event_side) else 0.0
    menu_loop_count = 1.0 if _is_menu_or_prompt_event(event_side) else 0.0
    death_count = 1.0 if event_side.get("death") else 0.0
    visible_map_novelty = float(_transition_visible_map_novelty_count(event_side))
    raw_progress_signal = reward > 0.0 or score_delta > 0.0 or depth_delta > 0.0
    dirty_progress_signal = raw_progress_signal and (
        wall_count > 0.0
        or bad_message_count > 0.0
        or non_advancing_count > 0.0
        or menu_loop_count > 0.0
        or death_count > 0.0
    )
    live_progress_event_count = (
        1.0 if raw_progress_signal and not dirty_progress_signal else 0.0
    )
    meaningful_event_count = float(
        _transition_meaningful_event_count(
            event_side=event_side,
            score_delta=score_delta,
            depth_delta=depth_delta,
            reward=reward,
            wall_count=wall_count,
            bad_message_count=bad_message_count,
            non_advancing_count=non_advancing_count,
            menu_loop_count=menu_loop_count,
        )
    )
    zero_progress_episode = (
        1.0 if not raw_progress_signal and live_progress_event_count == 0.0 else 0.0
    )
    visible_map_novelty_bonus_count = min(
        visible_map_novelty, float(MAX_VISIBLE_MAP_NOVELTY_BONUS_COUNT)
    )
    meaningful_event_bonus_count = min(
        meaningful_event_count, float(MAX_MEANINGFUL_EVENT_BONUS_COUNT)
    )
    components = {
        "normalized_score_delta_bonus": FITNESS_WEIGHTS["normalized_score_delta"]
        * _signed_log1p(score_delta),
        "cumulative_reward_bonus": FITNESS_WEIGHTS["cumulative_reward"]
        * _signed_log1p(reward),
        "depth_delta_bonus": FITNESS_WEIGHTS["depth_delta"] * depth_delta,
        "visible_map_novelty_bonus": FITNESS_WEIGHTS["visible_map_novelty"]
        * visible_map_novelty_bonus_count,
        "meaningful_event_bonus": FITNESS_WEIGHTS["meaningful_event"]
        * meaningful_event_bonus_count,
        "live_progress_event_bonus": FITNESS_WEIGHTS["live_progress_event"]
        * live_progress_event_count,
        "hp_damage_penalty": FITNESS_WEIGHTS["hp_damage"] * hp_damage,
        "wall_message_penalty": FITNESS_WEIGHTS["wall_or_solid_stone_message"]
        * wall_count,
        "wall_message_rate_penalty": FITNESS_WEIGHTS["wall_message_rate"] * wall_count,
        "bad_message_penalty": FITNESS_WEIGHTS["bad_message"] * bad_message_count,
        "bad_message_rate_penalty": FITNESS_WEIGHTS["bad_message_rate"]
        * bad_message_count,
        "non_advancing_step_penalty": FITNESS_WEIGHTS["non_advancing_step"]
        * non_advancing_count,
        "non_advancing_step_rate_penalty": FITNESS_WEIGHTS["non_advancing_step_rate"]
        * non_advancing_count,
        "menu_or_prompt_step_rate_penalty": FITNESS_WEIGHTS["menu_or_prompt_step_rate"]
        * menu_loop_count,
        "starvation_or_faint_penalty": FITNESS_WEIGHTS["starvation_or_faint"]
        * starvation_count,
        "stuck_menu_or_prompt_loop_penalty": FITNESS_WEIGHTS[
            "stuck_menu_or_prompt_loop"
        ]
        * menu_loop_count,
        "dirty_live_progress_event_penalty": FITNESS_WEIGHTS[
            "dirty_live_progress_event"
        ]
        * (1.0 if dirty_progress_signal else 0.0),
        "zero_progress_episode_penalty": FITNESS_WEIGHTS["zero_progress_episode"]
        * zero_progress_episode,
        "death_penalty": FITNESS_WEIGHTS["death"] * death_count,
    }
    positive_signal = bool(live_progress_event_count > 0.0)
    bad_signal = (
        hp_damage > 0.0
        or wall_count > 0.0
        or bad_message_count > 0.0
        or non_advancing_count > 0.0
        or starvation_count > 0.0
        or menu_loop_count > 0.0
        or death_count > 0.0
    )
    return TransitionUtilityBreakdown(
        value=sum(components.values()),
        components=components,
        positive_signal=positive_signal,
        bad_signal=bad_signal,
    )


def _is_high_quality_preference(
    chosen: TransitionUtilityBreakdown,
    rejected: TransitionUtilityBreakdown,
) -> bool:
    if chosen.positive_signal:
        return True
    if rejected.bad_signal and not chosen.bad_signal:
        return True
    return False


def _events_from_watch_report(watch_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = watch_report.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return []
    return [dict(event) for event in events if isinstance(event, Mapping)]


def _valid_action_ids(watch_report: Mapping[str, Any]) -> list[int]:
    action_manifest = watch_report.get("action_manifest")
    if not isinstance(action_manifest, Mapping):
        raise ValueError("watch report is missing action_manifest")
    raw_ids = action_manifest.get("valid_action_ids")
    if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
        raise ValueError("watch report action_manifest.valid_action_ids must be a list")
    action_ids = [int(action_id) for action_id in raw_ids]
    if not action_ids:
        raise ValueError("watch report action_manifest.valid_action_ids is empty")
    return action_ids


def _policy_prompt_text(
    event_side: Mapping[str, Any],
    *,
    valid_action_ids: Sequence[int],
) -> str | None:
    exact_prompt = event_side.get("policy_user_prompt")
    if isinstance(exact_prompt, str) and exact_prompt:
        return exact_prompt
    for key in ("policy_observation_text", "prompt_terminal_frame"):
        value = event_side.get(key)
        if isinstance(value, str) and value:
            return build_policy_prompt(
                observation_text=value,
                valid_action_ids=[int(action_id) for action_id in valid_action_ids],
                history=[],
                history_mode="single_frame",
            )
    return None


def _fitness_objective_version(watch_report: Mapping[str, Any]) -> str:
    rollout_metrics = watch_report.get("rollout_metrics")
    if isinstance(rollout_metrics, Mapping):
        for side_name in ("current", "baseline"):
            side = rollout_metrics.get(side_name)
            if isinstance(side, Mapping):
                version = side.get("fitness_objective_version")
                if isinstance(version, str) and version:
                    return version
    return FITNESS_OBJECTIVE_VERSION


def _delta(
    event_side: Mapping[str, Any],
    *,
    current_key: str,
    prompt_key: str,
) -> float:
    current = _float_or_none(event_side.get(current_key))
    prompt = _float_or_none(event_side.get(prompt_key))
    if current is None or prompt is None:
        return 0.0
    return current - prompt


def _signed_log1p(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _is_wall_message(message: str) -> bool:
    text = message.lower()
    return "wall" in text or "solid stone" in text


def _is_wall_event(event_side: Mapping[str, Any]) -> bool:
    return _is_wall_message(str(event_side.get("message") or "")) or _is_wall_message(
        str(event_side.get("terminal_frame") or "")
    )


def _is_bad_message_event(event_side: Mapping[str, Any]) -> bool:
    if _is_wall_event(event_side) or _is_menu_or_prompt_event(event_side):
        return True
    text = _event_status_text(event_side)
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


def _is_starvation_or_faint_event(event_side: Mapping[str, Any]) -> bool:
    text = _event_status_text(event_side)
    return "starv" in text or "faint" in text


def _is_menu_or_prompt_event(event_side: Mapping[str, Any]) -> bool:
    if event_side.get("menu_open") is True:
        return True
    text = _event_status_text(event_side)
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


def _event_status_text(event_side: Mapping[str, Any]) -> str:
    return " ".join(
        str(event_side.get(key) or "")
        for key in ("message", "hunger", "death", "terminal_frame")
    ).lower()


def _transition_visible_map_novelty_count(event_side: Mapping[str, Any]) -> int:
    prompt = event_side.get("prompt_terminal_frame")
    if not isinstance(prompt, str) or not prompt:
        prompt = event_side.get("policy_observation_text")
    terminal = event_side.get("terminal_frame")
    if not isinstance(prompt, str) or not isinstance(terminal, str):
        return 0
    return len(_visible_map_tile_keys(terminal) - _visible_map_tile_keys(prompt))


def _transition_meaningful_event_count(
    *,
    event_side: Mapping[str, Any],
    score_delta: float,
    depth_delta: float,
    reward: float,
    wall_count: float,
    bad_message_count: float,
    non_advancing_count: float,
    menu_loop_count: float,
) -> int:
    if event_side.get("action_id") is None:
        return 0
    if (
        wall_count > 0.0
        or bad_message_count > 0.0
        or non_advancing_count > 0.0
        or menu_loop_count > 0.0
    ):
        return 0
    if reward > 0.0 or score_delta > 0.0 or depth_delta > 0.0:
        return 1
    return 0


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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
