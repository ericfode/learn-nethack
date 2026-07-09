"""Shared policy-feedback prompt rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any


MAX_POLICY_FEEDBACK_ITEMS = 6


def build_policy_observation_with_feedback(
    *,
    observation_text: str,
    feedback_history: Sequence[Mapping[str, Any]],
) -> str:
    """Attach recent NLE action feedback before the rendered observation."""
    feedback = format_policy_feedback_history(feedback_history)
    if not feedback:
        return observation_text
    return "\n".join(
        [
            feedback,
            "",
            "Current rendered observation:",
            observation_text,
        ]
    )


def format_policy_feedback_history(
    feedback_history: Sequence[Mapping[str, Any]],
) -> str:
    """Render bounded action feedback for the next policy prompt."""
    if not feedback_history:
        return ""
    lines = ["Recent action feedback:"]
    for item in feedback_history[-MAX_POLICY_FEEDBACK_ITEMS:]:
        message = str(item.get("message") or "<missing>")
        lines.append(
            " ".join(
                [
                    f"- action_id={item.get('action_id')}",
                    f"reward={item.get('reward')}",
                    f"total={item.get('cumulative_reward')}",
                    f"hp={item.get('hp')}",
                    f"depth={item.get('depth')}",
                    f"advanced={item.get('game_time_advanced')}",
                    f"message={json.dumps(message)}",
                ]
            )
        )
    return "\n".join(lines)
