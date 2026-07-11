"""Counterfactual diagnostics for state-conditioned policy behavior."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
from typing import Any, Callable

from learn_nethack.compare_watch import select_action_id


SENSITIVITY_REPORT_SCHEMA_VERSION = "learn-nethack.policy-sensitivity.v1"
CURRENT_OBSERVATION_MARKER = "\nCurrent observation:\n"


@dataclass(frozen=True)
class PolicySensitivityCase:
    episode_id: str
    step: int
    target_action_id: int
    valid_action_ids: tuple[int, ...]
    natural_user_prompt: str
    shuffled_user_prompt: str
    natural_history: str
    shuffled_history: str
    natural_current_frame: str
    shuffled_current_frame: str
    shuffled_from_episode_id: str
    shuffled_from_step: int


def build_current_frame_shuffle_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[PolicySensitivityCase]:
    """Swap only current frames among comparable rows with distinct labels."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task") != "policy_action":
            continue
        grouped[_matching_key(row)].append(row)

    cases: list[PolicySensitivityCase] = []
    rng = random.Random(seed)
    for key in sorted(grouped, key=repr):
        group = sorted(grouped[key], key=_row_identity)
        rng.shuffle(group)
        assignments = _distinct_label_assignments(group)
        for natural, shuffled in assignments:
            natural_prompt = _user_prompt(natural)
            shuffled_prompt = _user_prompt(shuffled)
            natural_history, natural_frame = split_policy_user_prompt(natural_prompt)
            _shuffled_history, shuffled_frame = split_policy_user_prompt(
                shuffled_prompt
            )
            metadata = dict(natural.get("metadata") or {})
            cases.append(
                PolicySensitivityCase(
                    episode_id=str(natural.get("episode_id") or ""),
                    step=int(natural.get("step") or 0),
                    target_action_id=int(metadata["target_action_id"]),
                    valid_action_ids=tuple(
                        int(value) for value in metadata["valid_action_ids"]
                    ),
                    natural_user_prompt=natural_prompt,
                    shuffled_user_prompt=(
                        natural_history + CURRENT_OBSERVATION_MARKER + shuffled_frame
                    ),
                    natural_history=natural_history,
                    shuffled_history=natural_history,
                    natural_current_frame=natural_frame,
                    shuffled_current_frame=shuffled_frame,
                    shuffled_from_episode_id=str(
                        shuffled.get("episode_id") or ""
                    ),
                    shuffled_from_step=int(shuffled.get("step") or 0),
                )
            )
    return sorted(cases, key=lambda case: (case.episode_id, case.step))


def evaluate_policy_state_sensitivity(
    *,
    rows: Sequence[Mapping[str, Any]],
    policy: Any,
    seed: int,
    max_rows: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    selected_rows = list(rows[:max_rows] if max_rows is not None else rows)
    cases = build_current_frame_shuffle_cases(selected_rows, seed=seed)
    natural_matches = 0
    shuffled_matches = 0
    prediction_changes = 0
    samples: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases, start=1):
        natural_prediction = _predict_action(
            policy=policy,
            user_prompt=case.natural_user_prompt,
            valid_action_ids=case.valid_action_ids,
        )
        shuffled_prediction = _predict_action(
            policy=policy,
            user_prompt=case.shuffled_user_prompt,
            valid_action_ids=case.valid_action_ids,
        )
        natural_matches += int(natural_prediction == case.target_action_id)
        shuffled_matches += int(shuffled_prediction == case.target_action_id)
        prediction_changes += int(natural_prediction != shuffled_prediction)
        if len(samples) < 16:
            samples.append(
                {
                    "episode_id": case.episode_id,
                    "step": case.step,
                    "target_action_id": case.target_action_id,
                    "natural_prediction_action_id": natural_prediction,
                    "shuffled_prediction_action_id": shuffled_prediction,
                    "shuffled_from_episode_id": case.shuffled_from_episode_id,
                    "shuffled_from_step": case.shuffled_from_step,
                }
            )
        if progress_callback is not None and (
            case_index == 1 or case_index % 8 == 0 or case_index == len(cases)
        ):
            progress_callback(
                {
                    "phase": "policy_state_sensitivity",
                    "evaluated_cases": case_index,
                    "max_cases": len(cases),
                }
            )

    case_count = len(cases)
    natural_rate = _rate(natural_matches, case_count)
    shuffled_rate = _rate(shuffled_matches, case_count)
    comparable_group_count = len({_matching_key(row) for row in selected_rows})
    used_group_count = len(
        {
            _matching_key(row)
            for row in selected_rows
            if any(
                case.episode_id == str(row.get("episode_id") or "")
                and case.step == int(row.get("step") or 0)
                for case in cases
            )
        }
    )
    return {
        "schema_version": SENSITIVITY_REPORT_SCHEMA_VERSION,
        "seed": seed,
        "input_row_count": len(selected_rows),
        "case_count": case_count,
        "comparable_group_count": comparable_group_count,
        "used_group_count": used_group_count,
        "skipped_group_count": comparable_group_count - used_group_count,
        "natural_exact_match_rate": natural_rate,
        "shuffled_current_exact_match_rate": shuffled_rate,
        "current_state_dependence_gap": natural_rate - shuffled_rate,
        "prediction_change_rate_after_current_shuffle": _rate(
            prediction_changes,
            case_count,
        ),
        "samples": samples,
    }


def split_policy_user_prompt(user_prompt: str) -> tuple[str, str]:
    history, marker, current_frame = user_prompt.rpartition(
        CURRENT_OBSERVATION_MARKER
    )
    if not marker or not history or not current_frame:
        raise ValueError("policy user prompt is missing a final Current observation")
    return history, current_frame


def _matching_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = dict(row.get("metadata") or {})
    return (
        str(row.get("mode") or metadata.get("context_mode") or "unknown"),
        _context_length(row),
        str(metadata.get("role") or "unknown"),
        _depth_bucket(metadata.get("depth")),
    )


def _context_length(row: Mapping[str, Any]) -> int:
    metadata = dict(row.get("metadata") or {})
    recorded = metadata.get("context_item_count")
    if recorded is not None:
        return int(recorded)
    return _user_prompt(row).count("Previous action_id:")


def _depth_bucket(value: Any) -> str:
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return "unknown"
    if depth <= 1:
        return "1"
    if depth <= 3:
        return "2-3"
    if depth <= 6:
        return "4-6"
    return "7+"


def _distinct_label_assignments(
    group: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if len(group) < 2:
        return []
    assignments: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for index, natural in enumerate(group):
        natural_label = _target_action_id(natural)
        for offset in range(1, len(group)):
            candidate = group[(index + offset) % len(group)]
            if _target_action_id(candidate) != natural_label:
                assignments.append((natural, candidate))
                break
    return assignments


def _predict_action(
    *,
    policy: Any,
    user_prompt: str,
    valid_action_ids: Sequence[int],
) -> int:
    ids = [int(value) for value in valid_action_ids]
    scores = policy.score_actions(user_prompt=user_prompt, valid_action_ids=ids)
    return select_action_id(scores_by_action_id=scores, valid_action_ids=ids)


def _user_prompt(row: Mapping[str, Any]) -> str:
    messages = row.get("messages") or []
    prompts = [
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if not prompts:
        raise ValueError("policy row has no user prompt")
    return prompts[-1]


def _target_action_id(row: Mapping[str, Any]) -> int:
    return int(dict(row.get("metadata") or {})["target_action_id"])


def _row_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row.get("episode_id") or ""), int(row.get("step") or 0)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
