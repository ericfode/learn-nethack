"""SFT row builders for policy-action and next-frame tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from learn_nethack.observations import (
    observation_blstat,
    observation_message_text,
    render_observation_text,
)
from learn_nethack.policy_feedback import build_policy_observation_with_feedback


POLICY_SYSTEM_PROMPT = (
    'You control NetHack through NLE. Return only JSON: {"action_id": int}.'
)
NEXT_FRAME_SYSTEM_PROMPT = (
    "You predict NetHack transition dynamics from NLE traces. "
    "Return only the next rendered observation frame text. "
    "Begin with MAP: and include MESSAGE:, BLSTATS:, and INVENTORY: sections."
)


UNKNOWN_FEEDBACK_VALUE = "<unknown>"


@dataclass(frozen=True)
class HistoryEntry:
    observation_text: str
    action_id: int
    feedback: Mapping[str, Any] | None = None


class HistoryBuffer:
    def __init__(self, max_items: int = 16) -> None:
        self._items_by_gameid: dict[int, deque[HistoryEntry]] = defaultdict(
            lambda: deque(maxlen=max_items)
        )

    def append(
        self,
        *,
        gameid: int,
        observation_text: str,
        action_id: int,
        feedback: Mapping[str, Any] | None = None,
    ) -> None:
        self._items_by_gameid[int(gameid)].append(
            HistoryEntry(
                observation_text=observation_text,
                action_id=int(action_id),
                feedback=dict(feedback) if feedback is not None else None,
            )
        )

    def history_for(
        self,
        *,
        gameid: int,
        mode: str,
        token_budget: int,
    ) -> list[Any]:
        items = list(self._items_by_gameid.get(int(gameid), ()))
        if mode == "single_frame":
            return []
        if mode.startswith("context_"):
            count = int(mode.removeprefix("context_"))
            return _observation_history(items[-count:])
        if mode.startswith("feedback_context_"):
            count = int(mode.removeprefix("feedback_context_"))
            return _feedback_history(items[-count:])
        if mode == "growing_context":
            budget_chars = token_budget * 4
            selected = items[:]
            while selected and _history_char_count(selected) > budget_chars:
                selected.pop(0)
            return _observation_history(selected)
        if mode == "feedback_growing_context":
            budget_chars = token_budget * 4
            selected = items[:]
            while selected and _feedback_char_count(selected) > budget_chars:
                selected.pop(0)
            return _feedback_history(selected)
        raise ValueError(f"unknown SFT context mode: {mode}")


def _observation_history(history: Sequence[HistoryEntry]) -> list[tuple[str, int]]:
    return [(item.observation_text, item.action_id) for item in history]


def _feedback_history(history: Sequence[HistoryEntry]) -> list[Mapping[str, Any]]:
    return [dict(item.feedback) for item in history if item.feedback is not None]


def _history_char_count(history: Sequence[HistoryEntry]) -> int:
    return sum(
        len(item.observation_text) + len(str(item.action_id)) for item in history
    )


def _feedback_char_count(history: Sequence[HistoryEntry]) -> int:
    return sum(
        len(str(value))
        for item in history
        if item.feedback is not None
        for value in item.feedback.values()
    )


def policy_feedback_from_outcome_observation(
    *,
    action_id: int,
    observation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build offline feedback from the observation after a replay action."""
    if observation is None:
        return None
    return {
        "action_id": int(action_id),
        "reward": UNKNOWN_FEEDBACK_VALUE,
        "cumulative_reward": UNKNOWN_FEEDBACK_VALUE,
        "message": observation_message_text(observation),
        "hp": observation_blstat(observation, 10),
        "depth": observation_blstat(observation, 12),
        "game_time_advanced": UNKNOWN_FEEDBACK_VALUE,
    }


def _uses_feedback_history(mode: str) -> bool:
    return mode.startswith("feedback_context_") or mode == "feedback_growing_context"


def build_policy_prompt(
    *,
    observation_text: str,
    valid_action_ids: list[int],
    history: Sequence[Any],
    history_mode: str = "single_frame",
) -> str:
    lines = [f"Allowed action_ids: {valid_action_ids}"]
    if _uses_feedback_history(history_mode):
        lines.append("Current observation:")
        lines.append(
            build_policy_observation_with_feedback(
                observation_text=observation_text,
                feedback_history=[
                    item for item in history if isinstance(item, Mapping)
                ],
            )
        )
        return "\n".join(lines)
    if history:
        lines.append("Recent history:")
        for prior_observation, prior_action_id in history:
            lines.append(prior_observation)
            lines.append(f"Previous action_id: {prior_action_id}")
    lines.append("Current observation:")
    lines.append(observation_text)
    return "\n".join(lines)


def build_next_frame_prompt(
    *,
    observation_text: str,
    action_id: int,
    history: Sequence[Any],
    history_mode: str = "single_frame",
) -> str:
    lines = [f'Action taken: {{"action_id": {action_id}}}']
    if _uses_feedback_history(history_mode):
        lines.append("Current observation:")
        lines.append(
            build_policy_observation_with_feedback(
                observation_text=observation_text,
                feedback_history=[
                    item for item in history if isinstance(item, Mapping)
                ],
            )
        )
        return "\n".join(lines)
    if history:
        lines.append("Recent history:")
        for prior_observation, prior_action_id in history:
            lines.append(prior_observation)
            lines.append(f"Previous action_id: {prior_action_id}")
    lines.append("Current observation:")
    lines.append(observation_text)
    return "\n".join(lines)


def build_policy_action_row(
    *,
    dataset_name: str,
    split: str,
    mode: str,
    transition: Any,
    action_manifest: Any,
    game_metadata: dict[str, Any],
    history: Sequence[Any],
) -> dict[str, Any]:
    action_id = action_manifest.action_id_for_raw_key(transition.raw_key_code)
    observation_text = render_observation_text(transition.observation)
    user_prompt = build_policy_prompt(
        observation_text=observation_text,
        valid_action_ids=action_manifest.valid_action_ids(),
        history=history,
        history_mode=mode,
    )
    return {
        "schema_version": "learn-nethack.sft-row.v1",
        "dataset_name": dataset_name,
        "split": split,
        "task": "policy_action",
        "mode": mode,
        "gameid": transition.gameid,
        "episode_id": f"{dataset_name}:{transition.gameid}",
        "step": transition.step,
        **_sequence_fields(dataset_name=dataset_name, transition=transition),
        "messages": [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f'{{"action_id": {action_id}}}'},
        ],
        "metadata": {
            "target_action_id": action_id,
            "raw_key_code": transition.raw_key_code,
            "raw_key_label": action_manifest.key_label_for_raw_key(
                transition.raw_key_code
            ),
            "valid_action_ids": action_manifest.valid_action_ids(),
            "role": game_metadata.get("role"),
            "race": game_metadata.get("race"),
            "alignment": game_metadata.get("align"),
            "death": game_metadata.get("death"),
            "points": game_metadata.get("points"),
            "turns": game_metadata.get("turns"),
        },
    }


def build_pseudo_policy_action_row(
    *,
    dataset_name: str,
    split: str,
    mode: str,
    transition: Any,
    action_manifest: Any,
    game_metadata: dict[str, Any],
    history: Sequence[Any],
    pseudo_label: Any,
) -> dict[str, Any]:
    """Build a policy row from an explicit pseudo-label, not a true keypress."""
    action_id = int(pseudo_label.action_id)
    observation_text = render_observation_text(transition.observation)
    user_prompt = build_policy_prompt(
        observation_text=observation_text,
        valid_action_ids=action_manifest.valid_action_ids(),
        history=history,
        history_mode=mode,
    )
    return {
        "schema_version": "learn-nethack.sft-row.v1",
        "dataset_name": dataset_name,
        "split": split,
        "task": "policy_action",
        "mode": mode,
        "gameid": transition.gameid,
        "episode_id": f"{dataset_name}:{transition.gameid}",
        "step": transition.step,
        **_sequence_fields(dataset_name=dataset_name, transition=transition),
        "messages": [
            {"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f'{{"action_id": {action_id}}}'},
        ],
        "metadata": {
            "target_action_id": action_id,
            "valid_action_ids": action_manifest.valid_action_ids(),
            "label_source": pseudo_label.label_source,
            "label_confidence": pseudo_label.confidence,
            "label_reason": pseudo_label.reason,
            "pseudo_direction": pseudo_label.direction,
            "pseudo_action_name": pseudo_label.action_name,
            "pseudo_key_label": pseudo_label.key_label,
            "true_keypress_label_available": False,
            "role": game_metadata.get("role"),
            "race": game_metadata.get("race"),
            "alignment": game_metadata.get("align"),
            "death": game_metadata.get("death"),
            "points": game_metadata.get("points"),
            "turns": game_metadata.get("turns"),
        },
    }


def build_pseudo_next_frame_row(
    *,
    dataset_name: str,
    split: str,
    mode: str,
    transition: Any,
    action_manifest: Any,
    game_metadata: dict[str, Any],
    history: Sequence[Any],
    pseudo_label: Any,
    max_next_frame_chars: int = 4096,
) -> dict[str, Any]:
    """Build a next-frame row conditioned on an explicit pseudo action label."""
    if transition.next_observation is None:
        raise ValueError("missing_next_observation")
    action_id = int(pseudo_label.action_id)
    observation_text = render_observation_text(
        transition.observation,
        compact_map=True,
    )
    next_frame = render_observation_text(
        transition.next_observation,
        compact_map=True,
    )
    if len(next_frame) > max_next_frame_chars:
        next_frame = next_frame[:max_next_frame_chars]
    user_prompt = build_next_frame_prompt(
        observation_text=observation_text,
        action_id=action_id,
        history=history,
        history_mode=mode,
    )
    return {
        "schema_version": "learn-nethack.sft-row.v1",
        "dataset_name": dataset_name,
        "split": split,
        "task": "next_frame",
        "mode": mode,
        "gameid": transition.gameid,
        "episode_id": f"{dataset_name}:{transition.gameid}",
        "step": transition.step,
        **_sequence_fields(dataset_name=dataset_name, transition=transition),
        "messages": [
            {"role": "system", "content": NEXT_FRAME_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": next_frame},
        ],
        "metadata": {
            "conditioning_action_id": action_id,
            "valid_action_ids": action_manifest.valid_action_ids(),
            "label_source": pseudo_label.label_source,
            "label_confidence": pseudo_label.confidence,
            "label_reason": pseudo_label.reason,
            "pseudo_direction": pseudo_label.direction,
            "pseudo_action_name": pseudo_label.action_name,
            "pseudo_key_label": pseudo_label.key_label,
            "true_keypress_label_available": False,
            "target_frame_kind": "compact_rendered_observation_text",
            "next_frame_response_format": "raw_frame",
            "role": game_metadata.get("role"),
            "race": game_metadata.get("race"),
            "alignment": game_metadata.get("align"),
            "death": game_metadata.get("death"),
            "points": game_metadata.get("points"),
            "turns": game_metadata.get("turns"),
        },
    }


def build_next_frame_row(
    *,
    dataset_name: str,
    split: str,
    mode: str,
    transition: Any,
    action_manifest: Any,
    game_metadata: dict[str, Any],
    history: Sequence[Any],
    max_next_frame_chars: int = 4096,
) -> dict[str, Any]:
    if transition.next_observation is None:
        raise ValueError("missing_next_observation")
    action_id = action_manifest.action_id_for_raw_key(transition.raw_key_code)
    observation_text = render_observation_text(
        transition.observation,
        compact_map=True,
    )
    next_frame = render_observation_text(
        transition.next_observation,
        compact_map=True,
    )
    if len(next_frame) > max_next_frame_chars:
        next_frame = next_frame[:max_next_frame_chars]
    user_prompt = build_next_frame_prompt(
        observation_text=observation_text,
        action_id=action_id,
        history=history,
        history_mode=mode,
    )
    return {
        "schema_version": "learn-nethack.sft-row.v1",
        "dataset_name": dataset_name,
        "split": split,
        "task": "next_frame",
        "mode": mode,
        "gameid": transition.gameid,
        "episode_id": f"{dataset_name}:{transition.gameid}",
        "step": transition.step,
        **_sequence_fields(dataset_name=dataset_name, transition=transition),
        "messages": [
            {"role": "system", "content": NEXT_FRAME_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": next_frame},
        ],
        "metadata": {
            "conditioning_action_id": action_id,
            "raw_key_code": transition.raw_key_code,
            "raw_key_label": action_manifest.key_label_for_raw_key(
                transition.raw_key_code
            ),
            "target_frame_kind": "compact_rendered_observation_text",
            "next_frame_response_format": "raw_frame",
            "valid_action_ids": action_manifest.valid_action_ids(),
            "role": game_metadata.get("role"),
            "race": game_metadata.get("race"),
            "alignment": game_metadata.get("align"),
            "death": game_metadata.get("death"),
            "points": game_metadata.get("points"),
            "turns": game_metadata.get("turns"),
        },
    }


def _sequence_fields(*, dataset_name: str, transition: Any) -> dict[str, Any]:
    sequence_id = getattr(transition, "sequence_id", None)
    if not sequence_id:
        return {}
    sequence_step = getattr(transition, "sequence_step", None)
    return {
        "sequence_id": f"{dataset_name}:{sequence_id}",
        "sequence_step": (
            int(sequence_step) if sequence_step is not None else int(transition.step)
        ),
    }
