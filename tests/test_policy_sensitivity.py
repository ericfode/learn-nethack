from __future__ import annotations

from learn_nethack.policy_sensitivity import (
    build_current_frame_shuffle_cases,
    evaluate_policy_state_sensitivity,
)


def _row(*, step: int, frame: str, action_id: int) -> dict:
    return {
        "task": "policy_action",
        "mode": "context_1",
        "episode_id": f"episode-{step}",
        "step": step,
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": (
                    "Allowed action_ids: [0, 1]\n"
                    "Recent history:\n"
                    "MAP:\nprior\n"
                    "Previous action_id: 0\n"
                    "Current observation:\n"
                    f"MAP:\n{frame}\nMESSAGE:\n"
                ),
            },
            {"role": "assistant", "content": f'{{"action_id": {action_id}}}'},
        ],
        "metadata": {
            "target_action_id": action_id,
            "valid_action_ids": [0, 1],
            "role": "Arc",
            "depth": 1,
            "context_item_count": 1,
        },
    }


class FramePolicy:
    def score_actions(
        self,
        *,
        user_prompt: str,
        valid_action_ids: list[int],
    ) -> dict[int, float]:
        selected = 0 if "STATE_A" in user_prompt else 1
        return {
            action_id: 1.0 if action_id == selected else 0.0
            for action_id in valid_action_ids
        }


def test_shuffle_current_frames_preserves_history_and_labels() -> None:
    rows = [
        _row(step=1, frame="STATE_A", action_id=0),
        _row(step=2, frame="STATE_B", action_id=1),
    ]

    cases = build_current_frame_shuffle_cases(rows, seed=7)

    assert len(cases) == 2
    assert cases[0].target_action_id == rows[0]["metadata"]["target_action_id"]
    assert cases[0].natural_history == cases[0].shuffled_history
    assert cases[0].natural_current_frame != cases[0].shuffled_current_frame
    assert cases[0].valid_action_ids == (0, 1)


def test_state_sensitivity_reports_accuracy_drop_and_prediction_change() -> None:
    rows = [
        _row(step=1, frame="STATE_A", action_id=0),
        _row(step=2, frame="STATE_B", action_id=1),
    ]

    progress: list[dict] = []
    report = evaluate_policy_state_sensitivity(
        rows=rows,
        policy=FramePolicy(),
        seed=7,
        progress_callback=progress.append,
    )

    assert report["case_count"] == 2
    assert report["natural_exact_match_rate"] == 1.0
    assert report["shuffled_current_exact_match_rate"] == 0.0
    assert report["current_state_dependence_gap"] == 1.0
    assert report["prediction_change_rate_after_current_shuffle"] == 1.0
    assert report["skipped_group_count"] == 0
    assert progress[-1] == {
        "phase": "policy_state_sensitivity",
        "evaluated_cases": 2,
        "max_cases": 2,
    }


def test_shuffle_skips_groups_without_distinct_action_labels() -> None:
    rows = [
        _row(step=1, frame="STATE_A", action_id=0),
        _row(step=2, frame="STATE_B", action_id=0),
    ]

    report = evaluate_policy_state_sensitivity(
        rows=rows,
        policy=FramePolicy(),
        seed=7,
    )

    assert report["case_count"] == 0
    assert report["skipped_group_count"] == 1
