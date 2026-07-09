"""Held-out evaluation for matched local terminal world models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learn_nethack.action_manifest import load_action_manifest
from learn_nethack.world_model_metrics import (
    contiguous_rollout_starts,
    frame_metric_sums,
    per_example_changed_f1,
)
from learn_nethack.world_model_watch import terminal_text, write_world_model_watch


@dataclass(frozen=True)
class WorldModelEvalConfig:
    batch_size: int = 16
    one_step_examples: int = 256
    rollout_examples: int = 128
    action_ranking_examples: int = 256
    action_candidates: int = 8
    seed: int = 20260709


def build_rollout_starts(
    arrays: dict[str, np.ndarray],
    *,
    split_code: int,
    config: WorldModelEvalConfig,
) -> dict[int, np.ndarray]:
    indices = np.flatnonzero(arrays["split_codes"] == split_code)
    rng = np.random.default_rng(config.seed)
    shuffled = rng.permutation(indices)
    return {
        horizon: contiguous_rollout_starts(
            arrays["sequence_ids"],
            arrays["sequence_steps"],
            shuffled,
            horizon=horizon,
            limit=config.rollout_examples,
        )
        for horizon in (1, 5, 10)
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    arrays: dict[str, np.ndarray],
    test_indices: np.ndarray,
    train_indices: np.ndarray,
    rollout_starts: dict[int, np.ndarray],
    config: WorldModelEvalConfig,
    device,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    from learn_nethack._world_model_inference import candidate_action_scores
    from learn_nethack._world_model_torch import load_model_checkpoint, tensor_batch

    model, variant, checkpoint = load_model_checkpoint(
        checkpoint_path,
        device=device,
    )
    rng = np.random.default_rng(config.seed)
    one_step_indices = rng.permutation(test_indices)[: config.one_step_examples]
    one_step_prediction = _predict_one_step(
        model=model,
        variant=variant,
        arrays=arrays,
        indices=one_step_indices,
        batch_size=config.batch_size,
        device=device,
    )
    one_step_sums = frame_metric_sums(
        (
            arrays["current_chars"][one_step_indices],
            arrays["current_colors"][one_step_indices],
        ),
        (
            arrays["next_chars"][one_step_indices],
            arrays["next_colors"][one_step_indices],
        ),
        one_step_prediction,
    )
    rollout_report: dict[str, Any] = {}
    rollout_outputs: dict[int, dict[str, np.ndarray]] = {}
    for horizon, starts in rollout_starts.items():
        prediction = _predict_rollout(
            model=model,
            variant=variant,
            arrays=arrays,
            starts=starts,
            horizon=horizon,
            batch_size=config.batch_size,
            device=device,
        )
        truth_indices = starts + horizon - 1
        current = (
            arrays["current_chars"][starts],
            arrays["current_colors"][starts],
        )
        truth = (
            arrays["next_chars"][truth_indices],
            arrays["next_colors"][truth_indices],
        )
        sums = frame_metric_sums(current, truth, prediction)
        changed_f1 = per_example_changed_f1(current, truth, prediction)
        rollout_report[f"next_{horizon}"] = sums.as_metrics()
        rollout_outputs[horizon] = {
            "starts": starts,
            "predicted_chars": prediction[0],
            "predicted_colors": prediction[1],
            "changed_f1": changed_f1,
        }
    action_ranking = _evaluate_action_ranking(
        model=model,
        variant=variant,
        arrays=arrays,
        test_indices=test_indices,
        train_indices=train_indices,
        config=config,
        device=device,
        scorer=candidate_action_scores,
        tensor_batch=tensor_batch,
    )
    report = {
        "schema_version": "learn-nethack.local-world-model-eval.v1",
        "variant": variant,
        "checkpoint_path": str(checkpoint_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
        "one_step": one_step_sums.as_metrics(),
        "rollouts": rollout_report,
        "action_ranking": action_ranking,
        "inference_forward_passes_per_step": (
            1 if variant == "deterministic" else model.config.diffusion_steps
        ),
    }
    return report, rollout_outputs


def evaluate_copy_baseline(
    *,
    arrays: dict[str, np.ndarray],
    test_indices: np.ndarray,
    rollout_starts: dict[int, np.ndarray],
    config: WorldModelEvalConfig,
) -> dict[str, Any]:
    """Evaluate the trivial predictor that returns the current terminal."""
    rng = np.random.default_rng(config.seed)
    one_step_indices = rng.permutation(test_indices)[: config.one_step_examples]
    one_step_current = (
        arrays["current_chars"][one_step_indices],
        arrays["current_colors"][one_step_indices],
    )
    one_step = frame_metric_sums(
        one_step_current,
        (
            arrays["next_chars"][one_step_indices],
            arrays["next_colors"][one_step_indices],
        ),
        one_step_current,
    ).as_metrics()
    rollouts: dict[str, Any] = {}
    for horizon, starts in rollout_starts.items():
        truth_indices = starts + horizon - 1
        current = (
            arrays["current_chars"][starts],
            arrays["current_colors"][starts],
        )
        rollouts[f"next_{horizon}"] = frame_metric_sums(
            current,
            (
                arrays["next_chars"][truth_indices],
                arrays["next_colors"][truth_indices],
            ),
            current,
        ).as_metrics()
    return {
        "schema_version": "learn-nethack.local-world-model-eval.v1",
        "variant": "copy_current",
        "one_step": one_step,
        "rollouts": rollouts,
        "inference_forward_passes_per_step": 0,
    }


def write_comparison_watch(
    *,
    out_dir: str | Path,
    run_id: str,
    arrays: dict[str, np.ndarray],
    starts: np.ndarray,
    horizon: int,
    deterministic_outputs: dict[str, np.ndarray],
    diffusion_outputs: dict[str, np.ndarray],
    action_manifest_path: str | Path,
    limit: int = 16,
) -> dict[str, str | int]:
    labels = {
        entry.action_id: entry.key_label
        for entry in load_action_manifest(action_manifest_path).entries
    }
    events: list[dict[str, Any]] = []
    for position, start_value in enumerate(starts[:limit]):
        start = int(start_value)
        truth_index = start + horizon - 1
        final_action_index = truth_index
        events.append(
            {
                "schema_version": "learn-nethack.local-world-model-event.v1",
                "run_id": run_id,
                "event_index": position,
                "gameid": int(arrays["gameids"][start]),
                "sequence_step": int(arrays["sequence_steps"][start]),
                "horizon": horizon,
                "action_id": int(arrays["action_ids"][final_action_index]),
                "key_label": labels.get(
                    int(arrays["action_ids"][final_action_index]),
                    "unknown",
                ),
                "current_frame": terminal_text(arrays["current_chars"][start]),
                "ground_truth_frame": terminal_text(arrays["next_chars"][truth_index]),
                "deterministic_frame": terminal_text(
                    deterministic_outputs["predicted_chars"][position]
                ),
                "diffusion_frame": terminal_text(
                    diffusion_outputs["predicted_chars"][position]
                ),
                "baseline_changed_f1": float(
                    deterministic_outputs["changed_f1"][position]
                ),
                "diffusion_changed_f1": float(
                    diffusion_outputs["changed_f1"][position]
                ),
            }
        )
    return write_world_model_watch(out_dir=out_dir, run_id=run_id, events=events)


def _predict_one_step(
    *,
    model,
    variant: str,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    from learn_nethack._world_model_inference import decode_next_frame
    from learn_nethack._world_model_torch import tensor_batch

    predicted_chars: list[np.ndarray] = []
    predicted_colors: list[np.ndarray] = []
    for offset in range(0, indices.size, batch_size):
        selected = indices[offset : offset + batch_size]
        batch = tensor_batch(arrays, selected, device=device)
        chars, colors = decode_next_frame(
            model=model,
            variant=variant,
            current_chars=batch["current_chars"],
            current_colors=batch["current_colors"],
            action_ids=batch["action_ids"],
        )
        predicted_chars.append(chars.cpu().numpy().astype(np.uint8))
        predicted_colors.append(colors.cpu().numpy().astype(np.uint8))
    return np.concatenate(predicted_chars), np.concatenate(predicted_colors)


def _predict_rollout(
    *,
    model,
    variant: str,
    arrays: dict[str, np.ndarray],
    starts: np.ndarray,
    horizon: int,
    batch_size: int,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from learn_nethack._world_model_inference import decode_next_frame

    predicted_chars: list[np.ndarray] = []
    predicted_colors: list[np.ndarray] = []
    for offset in range(0, starts.size, batch_size):
        selected = starts[offset : offset + batch_size]
        current_chars = torch.as_tensor(
            arrays["current_chars"][selected],
            dtype=torch.long,
            device=device,
        )
        current_colors = torch.as_tensor(
            arrays["current_colors"][selected],
            dtype=torch.long,
            device=device,
        )
        for step in range(horizon):
            action_ids = torch.as_tensor(
                arrays["action_ids"][selected + step],
                dtype=torch.long,
                device=device,
            )
            current_chars, current_colors = decode_next_frame(
                model=model,
                variant=variant,
                current_chars=current_chars,
                current_colors=current_colors,
                action_ids=action_ids,
            )
        predicted_chars.append(current_chars.cpu().numpy().astype(np.uint8))
        predicted_colors.append(current_colors.cpu().numpy().astype(np.uint8))
    return np.concatenate(predicted_chars), np.concatenate(predicted_colors)


def _evaluate_action_ranking(
    *,
    model,
    variant: str,
    arrays: dict[str, np.ndarray],
    test_indices: np.ndarray,
    train_indices: np.ndarray,
    config: WorldModelEvalConfig,
    device,
    scorer,
    tensor_batch,
) -> dict[str, float | int]:
    import torch

    train_counts = Counter(int(value) for value in arrays["action_ids"][train_indices])
    common_actions = [action for action, _ in train_counts.most_common()]
    eligible = np.asarray(
        [
            index
            for index in test_indices
            if int(arrays["action_ids"][index]) in train_counts
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(config.seed + 17)
    selected = rng.permutation(eligible)[: config.action_ranking_examples]
    reciprocal_ranks: list[float] = []
    top_one = 0
    changed_reciprocal_ranks: list[float] = []
    changed_top_one = 0
    for offset in range(0, selected.size, config.batch_size):
        indices = selected[offset : offset + config.batch_size]
        candidates, true_positions = _candidate_matrix(
            true_actions=arrays["action_ids"][indices],
            common_actions=common_actions,
            candidate_count=config.action_candidates,
            rng=rng,
        )
        batch = tensor_batch(arrays, indices, device=device)
        scores = (
            scorer(
                model=model,
                variant=variant,
                current_chars=batch["current_chars"],
                current_colors=batch["current_colors"],
                next_chars=batch["next_chars"],
                next_colors=batch["next_colors"],
                candidate_actions=torch.as_tensor(
                    candidates,
                    dtype=torch.long,
                    device=device,
                ),
            )
            .cpu()
            .numpy()
        )
        changed = (
            arrays["current_chars"][indices] != arrays["next_chars"][indices]
        ).reshape(indices.size, -1).any(axis=1) | (
            arrays["current_colors"][indices] != arrays["next_colors"][indices]
        ).reshape(indices.size, -1).any(axis=1)
        for row, true_position in enumerate(true_positions):
            true_score = scores[row, true_position]
            rank = 1 + int((scores[row] >= true_score).sum()) - 1
            unique_top = bool(true_score > np.delete(scores[row], true_position).max())
            reciprocal_ranks.append(1.0 / rank)
            top_one += int(unique_top)
            if changed[row]:
                changed_reciprocal_ranks.append(1.0 / rank)
                changed_top_one += int(unique_top)
    return {
        "examples": len(reciprocal_ranks),
        "candidate_count": config.action_candidates,
        "top1_accuracy": _mean(top_one, len(reciprocal_ranks)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "changed_examples": len(changed_reciprocal_ranks),
        "changed_top1_accuracy": _mean(
            changed_top_one,
            len(changed_reciprocal_ranks),
        ),
        "changed_mean_reciprocal_rank": (
            float(np.mean(changed_reciprocal_ranks))
            if changed_reciprocal_ranks
            else 0.0
        ),
        "random_top1_accuracy": 1.0 / config.action_candidates,
        "random_mean_reciprocal_rank": float(
            sum(1.0 / rank for rank in range(1, config.action_candidates + 1))
            / config.action_candidates
        ),
        "skipped_true_action_unseen_in_train": int(test_indices.size - eligible.size),
    }


def _candidate_matrix(
    *,
    true_actions: np.ndarray,
    common_actions: list[int],
    candidate_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_count < 2:
        raise ValueError("action candidate count must be at least two")
    rows: list[list[int]] = []
    true_positions: list[int] = []
    for true_value in true_actions:
        true_action = int(true_value)
        candidates = [true_action]
        candidates.extend(action for action in common_actions if action != true_action)
        candidates = candidates[:candidate_count]
        if len(candidates) < candidate_count:
            raise ValueError("not enough observed actions for candidate ranking")
        rng.shuffle(candidates)
        rows.append(candidates)
        true_positions.append(candidates.index(true_action))
    return np.asarray(rows, dtype=np.int64), np.asarray(true_positions, dtype=np.int64)


def _mean(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / denominator
