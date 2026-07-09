"""Evaluation metrics for action and next-frame SFT tasks."""

from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
import math
from typing import Any, Callable

from learn_nethack.compare_watch import select_action_id
from learn_nethack.dynamics_play import parse_next_frame_response


NEXT_FRAME_PARSE_FAILURE_REASONS = (
    "invalid_json",
    "truncated_json",
    "wrong_schema",
    "wrong_value_type",
)
EXPECTED_WATCH_FITNESS_OBJECTIVE_VERSION = "live_rollout_utility_v7"
MIN_WATCH_PROOF_SEED_COUNT = 16
MIN_WATCH_CURRENT_FITNESS_SCORE = 0.0
MAX_WATCH_CURRENT_WALL_MESSAGE_RATE = 0.20
MAX_WATCH_CURRENT_BAD_MESSAGE_RATE = 0.20
MAX_WATCH_CURRENT_NON_ADVANCING_STEP_RATE = 0.20
MAX_WATCH_CURRENT_ACTION_REPEAT_RATE = 0.60
MAX_WATCH_CURRENT_MENU_OR_PROMPT_STEP_RATE = 0.20
MAX_WATCH_CURRENT_ZERO_PROGRESS_EPISODE_RATE = 0.0


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def compute_policy_metrics(
    *,
    predictions: list[Any],
    labels: list[int],
    valid_action_ids: set[int],
    metadata: list[dict[str, Any]],
) -> dict[str, float]:
    row_count = len(labels)
    parse_valid = 0
    action_space_valid = 0
    exact = 0
    role_totals: defaultdict[str, int] = defaultdict(int)
    role_exact: defaultdict[str, int] = defaultdict(int)

    for prediction, label, row_metadata in zip(predictions, labels, metadata):
        action_id = None
        if isinstance(prediction, dict) and isinstance(
            prediction.get("action_id"), int
        ):
            parse_valid += 1
            action_id = int(prediction["action_id"])
        if action_id in valid_action_ids:
            action_space_valid += 1
        if action_id == int(label):
            exact += 1
        role = str(row_metadata.get("role") or "unknown")
        role_totals[role] += 1
        if action_id == int(label):
            role_exact[role] += 1

    metrics = {
        "row_count": float(row_count),
        "parse_valid_rate": _rate(parse_valid, row_count),
        "action_space_valid_rate": _rate(action_space_valid, row_count),
        "exact_match_rate": _rate(exact, row_count),
    }
    for role, total in sorted(role_totals.items()):
        metrics[f"role_exact_match/{role}"] = _rate(role_exact[role], total)
    return metrics


def compute_next_frame_metrics(
    *,
    predictions: list[str],
    labels: list[str],
) -> dict[str, float]:
    row_count = len(labels)
    exact = sum(
        1 for prediction, label in zip(predictions, labels) if prediction == label
    )
    matching_chars = 0
    total_chars = 0
    for prediction, label in zip(predictions, labels):
        for pred_char, label_char in zip_longest(prediction, label, fillvalue=None):
            total_chars += 1
            if pred_char == label_char:
                matching_chars += 1
    return {
        "next_frame_eval_row_count": float(row_count),
        "next_frame_exact_match_rate": _rate(exact, row_count),
        "next_frame_char_accuracy": _rate(matching_chars, total_chars),
        "next_frame_map_line_exact_rate": _section_exact_rate(
            predictions, labels, "MAP:"
        ),
        "next_frame_message_exact_rate": _section_exact_rate(
            predictions, labels, "MESSAGE:"
        ),
    }


def evaluate_policy_rows_with_policy(
    *,
    rows: list[dict[str, Any]],
    policy: Any,
    max_rows: int | None = None,
) -> dict[str, float]:
    """Score policy SFT rows through constrained candidate-action selection."""
    predictions: list[dict[str, int]] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    valid_action_ids: set[int] = set()
    evaluated = 0
    for row in rows:
        if row.get("task") != "policy_action":
            continue
        if max_rows is not None and evaluated >= max_rows:
            break
        row_metadata = dict(row.get("metadata") or {})
        row_valid_action_ids = [
            int(value) for value in row_metadata["valid_action_ids"]
        ]
        scores = policy.score_actions(
            observation_text=_policy_observation_text(row),
            valid_action_ids=row_valid_action_ids,
        )
        predictions.append(
            {
                "action_id": select_action_id(
                    scores_by_action_id=scores,
                    valid_action_ids=row_valid_action_ids,
                )
            }
        )
        labels.append(int(row_metadata["target_action_id"]))
        metadata.append(row_metadata)
        valid_action_ids.update(row_valid_action_ids)
        evaluated += 1

    return compute_policy_metrics(
        predictions=predictions,
        labels=labels,
        valid_action_ids=valid_action_ids,
        metadata=metadata,
    )


def evaluate_next_frame_rows_with_predictor(
    *,
    rows: list[dict[str, Any]],
    predictor: Any,
    max_rows: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    sample_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    """Generate next-frame predictions from SFT rows and compute frame metrics."""
    predictions: list[str] = []
    labels: list[str] = []
    parse_valid = 0
    parse_failure_counts: defaultdict[str, int] = defaultdict(int)
    evaluated = 0
    for row in rows:
        if row.get("task") != "next_frame":
            continue
        if max_rows is not None and evaluated >= max_rows:
            break
        metadata = dict(row.get("metadata") or {})
        raw_prediction = predictor.generate_next_frame_json(
            observation_text=_current_observation_text(row),
            action_id=int(metadata["conditioning_action_id"]),
            history=[],
        )
        parser_error: str | None = None
        try:
            prediction = parse_next_frame_response(raw_prediction)
        except ValueError as exc:
            prediction = ""
            parser_error = str(exc)
            parse_failure_counts[
                _next_frame_parse_failure_reason(
                    raw_prediction=raw_prediction,
                    parser_error=parser_error,
                )
            ] += 1
        else:
            parse_valid += 1
        label = _next_frame_label(row)
        predictions.append(prediction)
        labels.append(label)
        evaluated += 1
        if sample_callback is not None:
            sample_callback(
                {
                    "phase": "next_frame_generate",
                    "row_index": evaluated - 1,
                    "action_id": int(metadata["conditioning_action_id"]),
                    "parse_valid": parser_error is None,
                    "parser_error": parser_error,
                    "parse_failure_reason": (
                        _next_frame_parse_failure_reason(
                            raw_prediction=raw_prediction,
                            parser_error=parser_error,
                        )
                        if parser_error is not None
                        else None
                    ),
                    "raw_output_chars": len(raw_prediction),
                    "label_chars": len(label),
                    "raw_output": _sample_text(raw_prediction),
                    "prediction": _sample_text(prediction),
                    "label": _sample_text(label),
                }
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "next_frame_generate",
                    "evaluated_rows": evaluated,
                    "max_rows": max_rows,
                    "parse_valid": parse_valid,
                }
            )

    metrics = compute_next_frame_metrics(predictions=predictions, labels=labels)
    metrics["next_frame_parse_valid_rate"] = _rate(parse_valid, evaluated)
    _add_next_frame_parse_failure_metrics(
        metrics=metrics,
        prefix="next_frame",
        counts=parse_failure_counts,
        total=evaluated,
    )
    return metrics


def evaluate_next_frame_sequences_with_predictor(
    *,
    rows: list[dict[str, Any]],
    predictor: Any,
    horizons: tuple[int, ...] = (1, 5, 10),
    max_windows: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    sample_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    """Autoregressively score next-n frame predictions over replay action windows."""
    sequence_rows = _episode_next_frame_rows(rows)
    metrics = summarize_next_frame_sequence_rows(
        rows=rows,
        horizons=horizons,
    )
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"next-frame sequence horizon must be positive: {horizon}")
        predictions: list[str] = []
        labels: list[str] = []
        parse_valid = 0
        parse_failure_counts: defaultdict[str, int] = defaultdict(int)
        windows = 0
        for episode_rows in sequence_rows:
            for start in range(0, len(episode_rows) - horizon + 1):
                if max_windows is not None and windows >= max_windows:
                    break
                current_frame = _current_observation_text(episode_rows[start])
                history: list[tuple[str, int]] = []
                for row in episode_rows[start : start + horizon]:
                    metadata = dict(row.get("metadata") or {})
                    action_id = int(metadata["conditioning_action_id"])
                    raw_prediction = predictor.generate_next_frame_json(
                        observation_text=current_frame,
                        action_id=action_id,
                        history=list(history),
                    )
                    parser_error: str | None = None
                    try:
                        prediction = parse_next_frame_response(raw_prediction)
                    except ValueError as exc:
                        prediction = ""
                        parser_error = str(exc)
                        parse_failure_counts[
                            _next_frame_parse_failure_reason(
                                raw_prediction=raw_prediction,
                                parser_error=parser_error,
                            )
                        ] += 1
                    else:
                        parse_valid += 1
                    label = _next_frame_label(row)
                    predictions.append(prediction)
                    labels.append(label)
                    if sample_callback is not None:
                        sample_callback(
                            {
                                "phase": "next_frame_sequence",
                                "horizon": horizon,
                                "window_index": windows,
                                "frame_index": len(labels) - 1,
                                "action_id": action_id,
                                "history_length": len(history),
                                "parse_valid": parser_error is None,
                                "parser_error": parser_error,
                                "parse_failure_reason": (
                                    _next_frame_parse_failure_reason(
                                        raw_prediction=raw_prediction,
                                        parser_error=parser_error,
                                    )
                                    if parser_error is not None
                                    else None
                                ),
                                "raw_output_chars": len(raw_prediction),
                                "label_chars": len(label),
                                "raw_output": _sample_text(raw_prediction),
                                "prediction": _sample_text(prediction),
                                "label": _sample_text(label),
                            }
                        )
                    history.append((current_frame, action_id))
                    current_frame = prediction
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "phase": "next_frame_sequence_frame",
                                "horizon": horizon,
                                "window_index": windows,
                                "generated_frames": len(labels),
                                "max_windows": max_windows,
                                "parse_valid": parse_valid,
                            }
                        )
                windows += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "next_frame_sequence",
                            "horizon": horizon,
                            "completed_windows": windows,
                            "max_windows": max_windows,
                            "generated_frames": len(labels),
                            "parse_valid": parse_valid,
                        }
                    )
            if max_windows is not None and windows >= max_windows:
                break
        frame_metrics = compute_next_frame_metrics(
            predictions=predictions,
            labels=labels,
        )
        prefix = f"next_{horizon}_frame_sequence"
        metrics[f"{prefix}_window_count"] = float(windows)
        metrics[f"{prefix}_frame_count"] = float(len(labels))
        metrics[f"{prefix}_parse_valid_rate"] = _rate(parse_valid, len(labels))
        _add_next_frame_parse_failure_metrics(
            metrics=metrics,
            prefix=prefix,
            counts=parse_failure_counts,
            total=len(labels),
        )
        metrics[f"{prefix}_exact_match_rate"] = frame_metrics[
            "next_frame_exact_match_rate"
        ]
        metrics[f"{prefix}_char_accuracy"] = frame_metrics["next_frame_char_accuracy"]
        metrics[f"{prefix}_map_line_exact_rate"] = frame_metrics[
            "next_frame_map_line_exact_rate"
        ]
        metrics[f"{prefix}_message_exact_rate"] = frame_metrics[
            "next_frame_message_exact_rate"
        ]
    return metrics


def summarize_next_frame_sequence_rows(
    *,
    rows: list[dict[str, Any]],
    horizons: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Report how much next-n frame evidence exists before model generation."""
    sequence_rows = _episode_next_frame_rows(rows)
    segment_lengths = [len(segment) for segment in sequence_rows]
    metrics = {
        "next_frame_sequence_row_count": float(sum(segment_lengths)),
        "next_frame_sequence_episode_count": float(
            len(
                {
                    str(row.get("episode_id") or row.get("gameid") or "unknown")
                    for segment in sequence_rows
                    for row in segment
                }
            )
        ),
        "next_frame_sequence_segment_count": float(len(segment_lengths)),
        "next_frame_sequence_max_segment_length": float(
            max(segment_lengths) if segment_lengths else 0
        ),
    }
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"next-frame sequence horizon must be positive: {horizon}")
        available_windows = sum(
            max(0, segment_length - horizon + 1) for segment_length in segment_lengths
        )
        eligible_segments = sum(
            1 for segment_length in segment_lengths if segment_length >= horizon
        )
        prefix = f"next_{horizon}_frame_sequence"
        metrics[f"{prefix}_available_window_count"] = float(available_windows)
        metrics[f"{prefix}_available_frame_count"] = float(available_windows * horizon)
        metrics[f"{prefix}_eligible_segment_count"] = float(eligible_segments)
    return metrics


def _sample_text(text: str, *, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n<truncated>"


def _next_frame_parse_failure_reason(
    *,
    raw_prediction: str,
    parser_error: str,
) -> str:
    if parser_error == "invalid next-frame JSON":
        stripped = raw_prediction.strip()
        if _looks_like_truncated_next_frame_json(stripped):
            return "truncated_json"
        return "invalid_json"
    if parser_error == "expected only next_frame in model output":
        return "wrong_schema"
    if parser_error == "next_frame must be a string":
        return "wrong_value_type"
    return "invalid_json"


def _looks_like_truncated_next_frame_json(stripped: str) -> bool:
    if not stripped.startswith("{") or '"next_frame"' not in stripped[:80]:
        return False
    return not stripped.endswith("}") or stripped.endswith("\\")


def _add_next_frame_parse_failure_metrics(
    *,
    metrics: dict[str, float],
    prefix: str,
    counts: dict[str, int],
    total: int,
) -> None:
    for reason in NEXT_FRAME_PARSE_FAILURE_REASONS:
        metrics[f"{prefix}_parse_failure_{reason}_rate"] = _rate(
            int(counts.get(reason, 0)),
            total,
        )


def evaluate_next_frame_rows_with_scorer(
    *,
    rows: list[dict[str, Any]],
    scorer: Any,
    max_rows: int | None = None,
) -> dict[str, float]:
    """Teacher-force next-frame labels and compute token-level likelihood metrics."""
    total_nll = 0.0
    total_tokens = 0
    total_matches = 0
    evaluated = 0
    for row in rows:
        if row.get("task") != "next_frame":
            continue
        if max_rows is not None and evaluated >= max_rows:
            break
        metadata = dict(row.get("metadata") or {})
        score = scorer.score_next_frame_response(
            observation_text=_current_observation_text(row),
            action_id=int(metadata["conditioning_action_id"]),
            target_response=_assistant_label_text(row),
            history=[],
        )
        token_count = int(score["token_count"])
        if token_count <= 0:
            continue
        total_nll += float(score["negative_log_likelihood"])
        total_tokens += token_count
        total_matches += int(score.get("argmax_match_count", 0))
        evaluated += 1

    mean_nll = total_nll / total_tokens if total_tokens else 0.0
    return {
        "next_frame_teacher_forced_row_count": float(evaluated),
        "next_frame_teacher_forced_token_count": float(total_tokens),
        "next_frame_teacher_forced_mean_nll": mean_nll,
        "next_frame_teacher_forced_perplexity": (
            math.exp(min(mean_nll, 100.0)) if total_tokens else 0.0
        ),
        "next_frame_teacher_forced_token_accuracy": _rate(
            total_matches,
            total_tokens,
        ),
    }


def _policy_observation_text(row: dict[str, Any]) -> str:
    return _current_observation_text(row)


def _current_observation_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    user_messages = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "user"
    ]
    if not user_messages:
        return ""
    content = user_messages[-1]
    marker = "Current observation:\n"
    if marker in content:
        return content.split(marker)[-1]
    return content


def _next_frame_label(row: dict[str, Any]) -> str:
    return parse_next_frame_response(_assistant_label_text(row))


def _assistant_label_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    assistant_messages = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    ]
    if not assistant_messages:
        raise ValueError("next_frame row has no assistant label")
    return assistant_messages[-1]


def _episode_next_frame_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    rows_by_episode: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task") != "next_frame":
            continue
        episode_id = str(
            row.get("sequence_id")
            or row.get("episode_id")
            or row.get("gameid")
            or "unknown"
        )
        rows_by_episode[episode_id].append(row)
    episodes: list[list[dict[str, Any]]] = []
    for episode_rows in rows_by_episode.values():
        sorted_rows = sorted(
            episode_rows,
            key=lambda row: int(row.get("sequence_step", row.get("step", 0))),
        )
        consecutive: list[dict[str, Any]] = []
        last_step: int | None = None
        for row in sorted_rows:
            step = int(row.get("sequence_step", row.get("step", 0)))
            if last_step is not None and step != last_step + 1:
                if consecutive:
                    episodes.append(consecutive)
                consecutive = []
            consecutive.append(row)
            last_step = step
        if consecutive:
            episodes.append(consecutive)
    return episodes


def _section_exact_rate(
    predictions: list[str], labels: list[str], header: str
) -> float:
    total = len(labels)
    exact = 0
    for prediction, label in zip(predictions, labels):
        if _section(prediction, header) == _section(label, header):
            exact += 1
    return _rate(exact, total)


def _section(text: str, header: str) -> str:
    lines = text.splitlines()
    try:
        start = lines.index(header) + 1
    except ValueError:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].endswith(":") and lines[index].isupper():
            end = index
            break
    return "\n".join(lines[start:end])


def build_score_to_beat_report(
    *,
    baseline_metrics: dict[str, float] | None,
    trained_metrics: dict[str, float],
    baseline_run_id: str | None,
    trained_run_id: str,
    metric_names: tuple[str, ...] = (
        "exact_match_rate",
        "action_space_valid_rate",
        "parse_valid_rate",
        "next_frame_parse_valid_rate",
        "next_frame_char_accuracy",
        "next_frame_exact_match_rate",
        "next_frame_map_line_exact_rate",
        "next_frame_message_exact_rate",
        "next_frame_teacher_forced_mean_nll",
        "next_frame_teacher_forced_perplexity",
        "next_frame_teacher_forced_token_accuracy",
        "next_1_frame_sequence_parse_valid_rate",
        "next_1_frame_sequence_available_window_count",
        "next_1_frame_sequence_available_frame_count",
        "next_1_frame_sequence_eligible_segment_count",
        "next_1_frame_sequence_window_count",
        "next_1_frame_sequence_frame_count",
        "next_1_frame_sequence_exact_match_rate",
        "next_1_frame_sequence_char_accuracy",
        "next_1_frame_sequence_map_line_exact_rate",
        "next_1_frame_sequence_message_exact_rate",
        "next_5_frame_sequence_parse_valid_rate",
        "next_5_frame_sequence_available_window_count",
        "next_5_frame_sequence_available_frame_count",
        "next_5_frame_sequence_eligible_segment_count",
        "next_5_frame_sequence_window_count",
        "next_5_frame_sequence_frame_count",
        "next_5_frame_sequence_exact_match_rate",
        "next_5_frame_sequence_char_accuracy",
        "next_5_frame_sequence_map_line_exact_rate",
        "next_5_frame_sequence_message_exact_rate",
        "next_10_frame_sequence_parse_valid_rate",
        "next_10_frame_sequence_available_window_count",
        "next_10_frame_sequence_available_frame_count",
        "next_10_frame_sequence_eligible_segment_count",
        "next_10_frame_sequence_window_count",
        "next_10_frame_sequence_frame_count",
        "next_10_frame_sequence_exact_match_rate",
        "next_10_frame_sequence_char_accuracy",
        "next_10_frame_sequence_map_line_exact_rate",
        "next_10_frame_sequence_message_exact_rate",
    ),
) -> dict[str, Any]:
    """Compare trained metrics against a baseline and return a proof report."""
    if baseline_metrics is None:
        return {
            "schema_version": "learn-nethack.score-to-beat.v1",
            "score_to_beat_status": "base_gemma_baseline_not_recorded",
            "baseline_run_id": baseline_run_id,
            "trained_run_id": trained_run_id,
            "verdict": "unproven",
            "metrics": {},
        }

    compared: dict[str, dict[str, float | bool]] = {}
    improved_count = 0
    regressed_count = 0
    for metric_name in metric_names:
        if metric_name not in baseline_metrics or metric_name not in trained_metrics:
            continue
        baseline_value = float(baseline_metrics[metric_name])
        trained_value = float(trained_metrics[metric_name])
        delta = round(trained_value - baseline_value, 12)
        direction = _metric_direction(metric_name)
        if direction == "lower_is_better":
            improved = delta < 0.0
            regressed = delta > 0.0
        else:
            improved = delta > 0.0
            regressed = delta < 0.0
        if improved:
            improved_count += 1
        if regressed:
            regressed_count += 1
        compared[metric_name] = {
            "baseline": baseline_value,
            "trained": trained_value,
            "delta": delta,
            "direction": direction,
            "improved": improved,
            "regressed": regressed,
        }

    proof_failures = _next_frame_sequence_proof_failures(
        baseline_metrics=baseline_metrics,
        trained_metrics=trained_metrics,
    )
    if proof_failures:
        verdict = "unproven"
    elif improved_count and not regressed_count:
        verdict = "improved"
    elif regressed_count and not improved_count:
        verdict = "regressed"
    elif improved_count or regressed_count:
        verdict = "mixed"
    else:
        verdict = "unchanged"

    return {
        "schema_version": "learn-nethack.score-to-beat.v1",
        "score_to_beat_status": "baseline_recorded",
        "baseline_run_id": baseline_run_id,
        "trained_run_id": trained_run_id,
        "verdict": verdict,
        "proof_failures": proof_failures,
        "metrics": compared,
    }


def build_training_proof_gate_report(
    *,
    score_to_beat_report: dict[str, Any] | None,
    watch_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine offline SFT and live watch evidence into one training verdict."""
    requirements: list[dict[str, Any]] = []
    score_metrics = (
        dict(score_to_beat_report.get("metrics") or {}) if score_to_beat_report else {}
    )
    rollout_metrics = (
        dict(watch_report.get("rollout_metrics") or {}) if watch_report else {}
    )
    watch_current = dict(rollout_metrics.get("current") or {})
    watch_baseline = dict(rollout_metrics.get("baseline") or {})
    watch_deltas = dict(rollout_metrics.get("deltas") or {})

    requirements.append(
        _proof_requirement(
            name="offline_score_to_beat_recorded",
            status=(
                "passed"
                if score_to_beat_report
                and score_to_beat_report.get("score_to_beat_status")
                == "baseline_recorded"
                else "missing"
            ),
            reason="offline baseline/trained score-to-beat report is required",
            evidence={
                "score_to_beat_status": (score_to_beat_report or {}).get(
                    "score_to_beat_status"
                ),
                "verdict": (score_to_beat_report or {}).get("verdict"),
            },
        )
    )
    for failure in (score_to_beat_report or {}).get("proof_failures") or []:
        requirements.append(
            _proof_requirement(
                name=f"offline_proof_failure_{failure.get('reason', 'unknown')}",
                status="failed",
                reason="offline score-to-beat report declared a proof failure",
                evidence=failure,
            )
        )

    requirements.extend(
        [
            _metric_requirement(
                score_metrics,
                "parse_valid_rate",
                mode="not_regressed",
                reason="policy output JSON parse validity must not regress",
            ),
            _metric_requirement(
                score_metrics,
                "action_space_valid_rate",
                mode="not_regressed",
                reason="policy action-space validity must not regress",
            ),
            _metric_requirement(
                score_metrics,
                "exact_match_rate",
                mode="not_regressed",
                reason="policy imitation must not be damaged by the training recipe",
            ),
            _metric_requirement(
                score_metrics,
                "next_frame_parse_valid_rate",
                mode="not_regressed",
                reason="generated next-frame parse validity must not regress",
            ),
        ]
    )
    for horizon in (1, 5, 10):
        prefix = f"next_{horizon}_frame_sequence"
        requirements.extend(
            [
                _metric_requirement(
                    score_metrics,
                    f"{prefix}_window_count",
                    mode="positive_trained_value",
                    reason=f"next-{horizon} sequence eval must include generated windows",
                ),
                _metric_requirement(
                    score_metrics,
                    f"{prefix}_parse_valid_rate",
                    mode="not_regressed",
                    reason=f"next-{horizon} generated sequence parse validity must not regress",
                ),
                _metric_requirement(
                    score_metrics,
                    f"{prefix}_char_accuracy",
                    mode="improved",
                    reason=f"next-{horizon} generated sequence character accuracy must improve",
                ),
                _metric_requirement(
                    score_metrics,
                    f"{prefix}_exact_match_rate",
                    mode="not_regressed",
                    reason=f"next-{horizon} exact frame match must not regress",
                ),
            ]
        )

    requirements.append(
        _proof_requirement(
            name="watch_report_recorded",
            status="passed" if watch_report else "missing",
            reason="live watch comparison report is required",
            evidence={
                "run_id": (watch_report or {}).get("run_id"),
                "status": (watch_report or {}).get("status"),
            },
        )
    )
    requirements.append(_watch_paired_initial_state_requirement(watch_report))
    requirements.append(_watch_seed_count_requirement(watch_report))
    requirements.append(
        _watch_fitness_version_requirement(
            watch_current=watch_current,
            watch_baseline=watch_baseline,
        )
    )
    requirements.extend(
        [
            _watch_current_metric_floor_requirement(
                watch_current,
                "fitness_score",
                minimum=MIN_WATCH_CURRENT_FITNESS_SCORE,
                reason="absolute live composite fitness must be positive",
            ),
            _watch_current_metric_ceiling_requirement(
                watch_current,
                "wall_message_rate",
                maximum=MAX_WATCH_CURRENT_WALL_MESSAGE_RATE,
                reason="current wall-message rate must stay below proof ceiling",
            ),
            _watch_current_metric_ceiling_requirement(
                watch_current,
                "bad_message_rate",
                maximum=MAX_WATCH_CURRENT_BAD_MESSAGE_RATE,
                reason="current bad prompt/action message rate must stay below proof ceiling",
            ),
            _watch_current_metric_ceiling_requirement(
                watch_current,
                "non_advancing_step_rate",
                maximum=MAX_WATCH_CURRENT_NON_ADVANCING_STEP_RATE,
                reason="current non-advancing step rate must stay below proof ceiling",
            ),
            _watch_current_metric_ceiling_requirement(
                watch_current,
                "action_repeat_rate",
                maximum=MAX_WATCH_CURRENT_ACTION_REPEAT_RATE,
                reason="current action repetition must stay below proof ceiling",
            ),
            _watch_current_metric_ceiling_requirement(
                watch_current,
                "menu_or_prompt_step_rate",
                maximum=MAX_WATCH_CURRENT_MENU_OR_PROMPT_STEP_RATE,
                reason="current menu/prompt step rate must stay below proof ceiling",
            ),
            _watch_current_metric_ceiling_requirement(
                watch_current,
                "zero_progress_episode",
                maximum=MAX_WATCH_CURRENT_ZERO_PROGRESS_EPISODE_RATE,
                reason="proof rollouts must not be zero-progress episodes",
            ),
            _watch_current_progress_requirement(watch_current=watch_current),
        ]
    )
    requirements.extend(
        [
            _watch_delta_requirement(
                watch_deltas,
                "fitness_score",
                mode="improved",
                reason="live composite fitness must improve",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "hp_damage_observed",
                mode="not_worse",
                reason="observed HP damage must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "wall_message_rate",
                mode="not_worse",
                reason="wall-message rate must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "bad_message_rate",
                mode="not_worse",
                reason="bad prompt/action message rate must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "non_advancing_step_rate",
                mode="not_worse",
                reason="non-advancing step rate must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "action_repeat_rate",
                mode="not_worse",
                reason="action collapse/repetition must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "starvation_or_faint_count",
                mode="not_worse",
                reason="starvation/fainting events must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "menu_or_prompt_step_rate",
                mode="not_worse",
                reason="menu/prompt step rate must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "stuck_menu_or_prompt_loop_count",
                mode="not_worse",
                reason="stuck menu/prompt loops must not worsen",
            ),
            _watch_delta_requirement(
                watch_deltas,
                "dirty_live_progress_event_count",
                mode="not_worse",
                reason="dirty live-progress events must not worsen",
            ),
            _watch_progress_requirement(
                watch_current=watch_current,
                watch_baseline=watch_baseline,
                watch_deltas=watch_deltas,
            ),
        ]
    )

    statuses = {str(requirement["status"]) for requirement in requirements}
    if "failed" in statuses:
        verdict = "failed"
    elif "missing" in statuses:
        verdict = "unproven"
    elif statuses == {"passed"}:
        verdict = "proved_improved"
    else:
        verdict = "failed"
    return {
        "schema_version": "learn-nethack.training-proof-gate.v1",
        "score_to_beat_run_ids": {
            "baseline": (score_to_beat_report or {}).get("baseline_run_id"),
            "trained": (score_to_beat_report or {}).get("trained_run_id"),
        },
        "watch_run_id": (watch_report or {}).get("run_id"),
        "verdict": verdict,
        "passed": verdict == "proved_improved",
        "requirements": requirements,
    }


def _metric_direction(metric_name: str) -> str:
    lower_is_better = {
        "next_frame_teacher_forced_mean_nll",
        "next_frame_teacher_forced_perplexity",
    }
    if metric_name.endswith("_window_count") or metric_name.endswith("_frame_count"):
        return "higher_is_better"
    if metric_name in lower_is_better:
        return "lower_is_better"
    return "higher_is_better"


def _proof_requirement(
    *,
    name: str,
    status: str,
    reason: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "passed": status == "passed",
        "reason": reason,
        "evidence": evidence,
    }


def _metric_requirement(
    metrics: dict[str, Any],
    metric_name: str,
    *,
    mode: str,
    reason: str,
) -> dict[str, Any]:
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict):
        return _proof_requirement(
            name=metric_name,
            status="missing",
            reason=reason,
            evidence={"metric": metric_name},
        )
    baseline = metric.get("baseline")
    trained = metric.get("trained")
    delta = metric.get("delta")
    if mode == "improved":
        passed = bool(metric.get("improved"))
    elif mode == "not_regressed":
        passed = not bool(metric.get("regressed"))
    elif mode == "positive_trained_value":
        passed = _float_or_none(trained) is not None and float(trained) > 0.0
    else:
        raise ValueError(f"unknown metric proof mode: {mode!r}")
    return _proof_requirement(
        name=metric_name,
        status="passed" if passed else "failed",
        reason=reason,
        evidence={
            "baseline": baseline,
            "trained": trained,
            "delta": delta,
            "mode": mode,
            "direction": metric.get("direction"),
        },
    )


def _watch_delta_requirement(
    deltas: dict[str, Any],
    metric_name: str,
    *,
    mode: str,
    reason: str,
) -> dict[str, Any]:
    value = _float_or_none(deltas.get(metric_name))
    if value is None:
        return _proof_requirement(
            name=f"watch_{metric_name}",
            status="missing",
            reason=reason,
            evidence={"metric": metric_name},
        )
    if mode == "improved":
        passed = value > 0.0
    elif mode == "not_worse":
        passed = value <= 0.0
    else:
        raise ValueError(f"unknown watch proof mode: {mode!r}")
    return _proof_requirement(
        name=f"watch_{metric_name}",
        status="passed" if passed else "failed",
        reason=reason,
        evidence={"delta": value, "mode": mode},
    )


def _watch_paired_initial_state_requirement(
    watch_report: dict[str, Any] | None,
) -> dict[str, Any]:
    value = (watch_report or {}).get("paired_initial_state_equal")
    seed_count = _int_or_none((watch_report or {}).get("seed_count"))
    paired_count = _int_or_none(
        (watch_report or {}).get("paired_initial_state_equal_count")
    )
    if value is None and seed_count is not None and paired_count is not None:
        passed = seed_count > 0 and paired_count == seed_count
        return _proof_requirement(
            name="watch_paired_initial_state_equal",
            status="passed" if passed else "failed",
            reason="live watch baseline/current rollouts must start from the same rendered state",
            evidence={
                "paired_initial_state_equal": None,
                "paired_initial_state_equal_count": paired_count,
                "seed_count": seed_count,
            },
        )
    if value is None:
        return _proof_requirement(
            name="watch_paired_initial_state_equal",
            status="missing",
            reason="live watch baseline/current rollouts must start from the same rendered state",
            evidence={
                "paired_initial_state_equal": value,
                "paired_initial_state_equal_count": paired_count,
                "seed_count": seed_count,
            },
        )
    passed = bool(value)
    return _proof_requirement(
        name="watch_paired_initial_state_equal",
        status="passed" if passed else "failed",
        reason="live watch baseline/current rollouts must start from the same rendered state",
        evidence={
            "paired_initial_state_equal": value,
            "paired_initial_state_equal_count": paired_count,
            "seed_count": seed_count,
        },
    )


def _watch_seed_count_requirement(
    watch_report: dict[str, Any] | None,
) -> dict[str, Any]:
    seed_count = _int_or_none((watch_report or {}).get("seed_count"))
    inferred_seed_count = (
        1
        if seed_count is None
        and watch_report
        and watch_report.get("paired_initial_state_equal") is not None
        else seed_count
    )
    if inferred_seed_count is None:
        return _proof_requirement(
            name="watch_seed_count",
            status="missing",
            reason="live proof runs must include enough seeded watch episodes",
            evidence={
                "seed_count": seed_count,
                "minimum_seed_count": MIN_WATCH_PROOF_SEED_COUNT,
            },
        )
    passed = inferred_seed_count >= MIN_WATCH_PROOF_SEED_COUNT
    return _proof_requirement(
        name="watch_seed_count",
        status="passed" if passed else "failed",
        reason="live proof runs must include enough seeded watch episodes",
        evidence={
            "seed_count": inferred_seed_count,
            "reported_seed_count": seed_count,
            "minimum_seed_count": MIN_WATCH_PROOF_SEED_COUNT,
        },
    )


def _watch_fitness_version_requirement(
    *,
    watch_current: dict[str, Any],
    watch_baseline: dict[str, Any],
) -> dict[str, Any]:
    current_version = watch_current.get("fitness_objective_version")
    baseline_version = watch_baseline.get("fitness_objective_version")
    evidence = {
        "current_fitness_objective_version": current_version,
        "baseline_fitness_objective_version": baseline_version,
        "expected": EXPECTED_WATCH_FITNESS_OBJECTIVE_VERSION,
    }
    if current_version is None or baseline_version is None:
        return _proof_requirement(
            name="watch_fitness_objective_version",
            status="missing",
            reason="live watch fitness must use the versioned ML-analysis objective",
            evidence=evidence,
        )
    passed = (
        current_version == EXPECTED_WATCH_FITNESS_OBJECTIVE_VERSION
        and baseline_version == EXPECTED_WATCH_FITNESS_OBJECTIVE_VERSION
    )
    return _proof_requirement(
        name="watch_fitness_objective_version",
        status="passed" if passed else "failed",
        reason="live watch fitness must use the versioned ML-analysis objective",
        evidence=evidence,
    )


def _watch_current_metric_floor_requirement(
    watch_current: dict[str, Any],
    metric_name: str,
    *,
    minimum: float,
    reason: str,
) -> dict[str, Any]:
    value = _float_or_none(watch_current.get(metric_name))
    if value is None:
        return _proof_requirement(
            name=f"watch_current_{metric_name}_floor",
            status="missing",
            reason=reason,
            evidence={"current": None, "minimum": minimum},
        )
    passed = value > minimum
    return _proof_requirement(
        name=f"watch_current_{metric_name}_floor",
        status="passed" if passed else "failed",
        reason=reason,
        evidence={"current": value, "minimum": minimum},
    )


def _watch_current_metric_ceiling_requirement(
    watch_current: dict[str, Any],
    metric_name: str,
    *,
    maximum: float,
    reason: str,
) -> dict[str, Any]:
    value = _float_or_none(watch_current.get(metric_name))
    if value is None:
        return _proof_requirement(
            name=f"watch_current_{metric_name}_ceiling",
            status="missing",
            reason=reason,
            evidence={"current": None, "maximum": maximum},
        )
    passed = value <= maximum
    return _proof_requirement(
        name=f"watch_current_{metric_name}_ceiling",
        status="passed" if passed else "failed",
        reason=reason,
        evidence={"current": value, "maximum": maximum},
    )


def _watch_current_progress_requirement(
    *,
    watch_current: dict[str, Any],
) -> dict[str, Any]:
    score_delta = _float_or_none(watch_current.get("score_delta"))
    reward = _float_or_none(watch_current.get("cumulative_reward"))
    depth_delta = _float_or_none(watch_current.get("depth_delta"))
    if reward is None and score_delta is None and depth_delta is None:
        return _proof_requirement(
            name="watch_current_score_or_depth_progress",
            status="missing",
            reason="current live rollout must show absolute score/reward or depth progress",
            evidence={
                "current_score_delta": score_delta,
                "current_reward": reward,
                "current_depth_delta": depth_delta,
            },
        )
    passed = any(
        value is not None and value > 0.0
        for value in (score_delta, reward, depth_delta)
    )
    return _proof_requirement(
        name="watch_current_score_or_depth_progress",
        status="passed" if passed else "failed",
        reason="current live rollout must show absolute score/reward or depth progress",
        evidence={
            "current_score_delta": score_delta,
            "current_reward": reward,
            "current_depth_delta": depth_delta,
        },
    )


def _watch_progress_requirement(
    *,
    watch_current: dict[str, Any],
    watch_baseline: dict[str, Any],
    watch_deltas: dict[str, Any],
) -> dict[str, Any]:
    score_delta = _float_or_none(watch_deltas.get("score_delta"))
    reward_delta = _float_or_none(watch_deltas.get("cumulative_reward"))
    depth_delta = _float_or_none(watch_deltas.get("depth_delta"))
    if depth_delta is None:
        depth_delta = _float_or_none(watch_deltas.get("depth_max"))
    if reward_delta is None and score_delta is None:
        progress_delta = None
    else:
        progress_delta = max(
            value for value in (score_delta, reward_delta) if value is not None
        )
    if progress_delta is None or depth_delta is None:
        return _proof_requirement(
            name="watch_score_or_depth_progress",
            status="missing",
            reason="live rollout must report score/reward and depth deltas",
            evidence={
                "score_delta": score_delta,
                "reward_delta": reward_delta,
                "depth_delta": depth_delta,
            },
        )
    passed = progress_delta > 0.0 or depth_delta > 0.0
    return _proof_requirement(
        name="watch_score_or_depth_progress",
        status="passed" if passed else "failed",
        reason="live rollout must improve score/reward or dungeon depth",
        evidence={
            "score_delta": score_delta,
            "reward_delta": reward_delta,
            "depth_delta": depth_delta,
            "current_score_delta": watch_current.get("score_delta"),
            "baseline_score_delta": watch_baseline.get("score_delta"),
            "current_reward": watch_current.get("cumulative_reward"),
            "baseline_reward": watch_baseline.get("cumulative_reward"),
            "current_depth_delta": watch_current.get("depth_delta"),
            "baseline_depth_delta": watch_baseline.get("depth_delta"),
        },
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _next_frame_sequence_proof_failures(
    *,
    baseline_metrics: dict[str, float],
    trained_metrics: dict[str, float],
    horizons: tuple[int, ...] = (1, 5, 10),
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for horizon in horizons:
        prefix = f"next_{horizon}_frame_sequence"
        has_sequence_metric = any(
            key.startswith(prefix)
            for metrics in (baseline_metrics, trained_metrics)
            for key in metrics
        )
        if not has_sequence_metric:
            continue
        for run_label, metrics in (
            ("baseline", baseline_metrics),
            ("trained", trained_metrics),
        ):
            for suffix in ("available_window_count", "available_frame_count"):
                metric_name = f"{prefix}_{suffix}"
                value = metrics.get(metric_name)
                if value is not None and float(value) <= 0.0:
                    failures.append(
                        {
                            "run": run_label,
                            "horizon": horizon,
                            "metric": metric_name,
                            "value": float(value),
                            "reason": "zero_available_sequence_evidence",
                        }
                    )
            for suffix in ("window_count", "frame_count"):
                metric_name = f"{prefix}_{suffix}"
                value = metrics.get(metric_name)
                if value is None:
                    failures.append(
                        {
                            "run": run_label,
                            "horizon": horizon,
                            "metric": metric_name,
                            "reason": "missing_sequence_count",
                        }
                    )
                elif float(value) <= 0.0:
                    failures.append(
                        {
                            "run": run_label,
                            "horizon": horizon,
                            "metric": metric_name,
                            "value": float(value),
                            "reason": "zero_sequence_evidence",
                        }
                    )
    return failures
