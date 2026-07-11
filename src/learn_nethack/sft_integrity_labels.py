"""Row-level label checks used by the SFT dataset integrity audit."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_FRAME_SECTIONS = ("MAP:", "MESSAGE:", "BLSTATS:", "INVENTORY:")


def audit_policy_label(
    *,
    messages: list[Any],
    metadata: Mapping[str, Any],
    mapped_action_id: int | None,
    valid_action_id_set: set[int],
    path: Path,
    line_number: int,
    fail: Any,
) -> int | None:
    target_action_id = integer(metadata.get("target_action_id"))
    assistant_content = messages[-1].get("content")
    try:
        assistant_payload = json.loads(assistant_content)
    except (TypeError, json.JSONDecodeError):
        assistant_payload = None
    assistant_action_id = (
        integer(assistant_payload.get("action_id"))
        if isinstance(assistant_payload, Mapping)
        and set(assistant_payload) == {"action_id"}
        else None
    )
    if target_action_id is None or assistant_action_id is None:
        fail(
            "policy_action_json_invalid",
            path=str(path),
            line_number=line_number,
        )
        return target_action_id
    if target_action_id not in valid_action_id_set:
        fail(
            "policy_target_out_of_space",
            path=str(path),
            line_number=line_number,
            target_action_id=target_action_id,
        )
    if assistant_action_id != target_action_id:
        fail(
            "policy_assistant_target_mismatch",
            path=str(path),
            line_number=line_number,
            assistant_action_id=assistant_action_id,
            target_action_id=target_action_id,
        )
    if mapped_action_id is not None and target_action_id != mapped_action_id:
        fail(
            "policy_raw_key_target_mismatch",
            path=str(path),
            line_number=line_number,
            mapped_action_id=mapped_action_id,
            target_action_id=target_action_id,
        )
    return target_action_id


def audit_next_frame_label(
    *,
    messages: list[Any],
    metadata: Mapping[str, Any],
    mapped_action_id: int | None,
    valid_action_id_set: set[int],
    path: Path,
    line_number: int,
    fail: Any,
) -> tuple[int | None, bool | None]:
    action_id = integer(metadata.get("conditioning_action_id"))
    if action_id is None:
        fail(
            "next_frame_conditioning_action_missing",
            path=str(path),
            line_number=line_number,
        )
        return None, None
    if action_id not in valid_action_id_set:
        fail(
            "next_frame_action_out_of_space",
            path=str(path),
            line_number=line_number,
            action_id=action_id,
        )
    if mapped_action_id is not None and action_id != mapped_action_id:
        fail(
            "next_frame_raw_key_action_mismatch",
            path=str(path),
            line_number=line_number,
            mapped_action_id=mapped_action_id,
            action_id=action_id,
        )
    user_content = messages[-2].get("content")
    expected_prefix = f'Action taken: {{"action_id": {action_id}}}\n'
    if not isinstance(user_content, str) or not user_content.startswith(
        expected_prefix
    ):
        fail(
            "next_frame_prompt_action_mismatch",
            path=str(path),
            line_number=line_number,
            action_id=action_id,
        )
    assistant_content = messages[-1].get("content")
    if not isinstance(assistant_content, str) or any(
        section not in assistant_content for section in REQUIRED_FRAME_SECTIONS
    ):
        fail(
            "next_frame_target_shape_invalid",
            path=str(path),
            line_number=line_number,
        )
    marker = "\nCurrent observation:\n"
    if not isinstance(user_content, str) or marker not in user_content:
        return action_id, None
    current_frame = user_content.rsplit(marker, maxsplit=1)[-1]
    if not isinstance(assistant_content, str):
        return action_id, None
    return action_id, current_frame != assistant_content


def messages_have_final_assistant(messages: Any) -> bool:
    return (
        isinstance(messages, list)
        and len(messages) == 3
        and all(isinstance(message, Mapping) for message in messages)
        and [message.get("role") for message in messages]
        == ["system", "user", "assistant"]
        and all(isinstance(message.get("content"), str) for message in messages)
    )


def row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("episode_id"),
        row.get("step"),
        row.get("sequence_id"),
        row.get("sequence_step"),
    )


def action_distribution(
    histogram: Counter[int],
    *,
    action_names: Mapping[int, str],
) -> dict[str, Any]:
    total = sum(histogram.values())
    ordered = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))
    return {
        "total": total,
        "unique_action_count": len(histogram),
        "dominant_action_id": ordered[0][0] if ordered else None,
        "dominant_action_name": action_names.get(ordered[0][0]) if ordered else None,
        "dominant_action_rate": rate(ordered[0][1], total) if ordered else None,
        "histogram": {str(action_id): count for action_id, count in ordered},
    }


def count_jsonl_rows(path: Path, *, fail: Any) -> int:
    if not path.exists():
        fail("combined_file_missing", path=str(path))
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
