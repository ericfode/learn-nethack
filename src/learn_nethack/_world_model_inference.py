"""Optional PyTorch decoding and action scoring for local world models."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from learn_nethack._world_model_torch import TerminalDeltaModel, delta_targets
from learn_nethack.world_model_metrics import (
    CHAR_MASK_TOKEN,
    COLOR_MASK_TOKEN,
)


@torch.inference_mode()
def decode_next_frame(
    *,
    model: TerminalDeltaModel,
    variant: str,
    current_chars: torch.Tensor,
    current_colors: torch.Tensor,
    action_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = model.config
    batch_size, rows, cols = current_chars.shape
    char_delta = torch.full(
        (batch_size, rows, cols),
        CHAR_MASK_TOKEN,
        dtype=torch.long,
        device=current_chars.device,
    )
    color_delta = torch.full(
        (batch_size, rows, cols),
        COLOR_MASK_TOKEN,
        dtype=torch.long,
        device=current_chars.device,
    )
    if variant == "deterministic":
        timesteps = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=current_chars.device,
        )
        char_logits, color_logits = model(
            current_chars,
            current_colors,
            char_delta,
            color_delta,
            action_ids,
            timesteps,
        )
        char_delta = char_logits.argmax(dim=1)
        color_delta = color_logits.argmax(dim=1)
    elif variant == "diffusion":
        masked = torch.ones_like(char_delta, dtype=torch.bool)
        cell_count = rows * cols
        for timestep in range(config.diffusion_steps, 0, -1):
            timesteps = torch.full(
                (batch_size,),
                timestep,
                dtype=torch.long,
                device=current_chars.device,
            )
            char_logits, color_logits = model(
                current_chars,
                current_colors,
                char_delta,
                color_delta,
                action_ids,
                timesteps,
            )
            char_probability = char_logits.softmax(dim=1)
            color_probability = color_logits.softmax(dim=1)
            char_confidence, char_prediction = char_probability.max(dim=1)
            color_confidence, color_prediction = color_probability.max(dim=1)
            confidence = (char_confidence + color_confidence) / 2.0
            remaining = math.floor(cell_count * (timestep - 1) / config.diffusion_steps)
            _reveal_high_confidence(
                char_delta=char_delta,
                color_delta=color_delta,
                masked=masked,
                char_prediction=char_prediction,
                color_prediction=color_prediction,
                confidence=confidence,
                remaining=remaining,
            )
    else:
        raise ValueError(f"unsupported world-model variant: {variant}")
    next_chars = torch.where(char_delta == 0, current_chars, char_delta - 1)
    next_colors = torch.where(color_delta == 0, current_colors, color_delta - 1)
    return next_chars, next_colors


@torch.inference_mode()
def candidate_action_scores(
    *,
    model: TerminalDeltaModel,
    variant: str,
    current_chars: torch.Tensor,
    current_colors: torch.Tensor,
    next_chars: torch.Tensor,
    next_colors: torch.Tensor,
    candidate_actions: torch.Tensor,
) -> torch.Tensor:
    """Return a comparable full-noise denoising score for each candidate action."""
    batch_size, candidate_count = candidate_actions.shape
    rows, cols = current_chars.shape[1:]
    repeated = {
        "current_chars": current_chars[:, None]
        .expand(-1, candidate_count, -1, -1)
        .reshape(-1, rows, cols),
        "current_colors": current_colors[:, None]
        .expand(-1, candidate_count, -1, -1)
        .reshape(-1, rows, cols),
        "next_chars": next_chars[:, None]
        .expand(-1, candidate_count, -1, -1)
        .reshape(-1, rows, cols),
        "next_colors": next_colors[:, None]
        .expand(-1, candidate_count, -1, -1)
        .reshape(-1, rows, cols),
        "action_ids": candidate_actions.reshape(-1),
    }
    char_target, color_target = delta_targets(repeated)
    char_noisy = torch.full_like(char_target, CHAR_MASK_TOKEN)
    color_noisy = torch.full_like(color_target, COLOR_MASK_TOKEN)
    timestep = 0 if variant == "deterministic" else model.config.diffusion_steps
    timesteps = torch.full(
        (batch_size * candidate_count,),
        timestep,
        dtype=torch.long,
        device=current_chars.device,
    )
    char_logits, color_logits = model(
        repeated["current_chars"],
        repeated["current_colors"],
        char_noisy,
        color_noisy,
        repeated["action_ids"],
        timesteps,
    )
    char_loss = _per_example_categorical_loss(
        char_logits,
        char_target,
        change_weight=model.config.change_weight,
    )
    color_loss = _per_example_categorical_loss(
        color_logits,
        color_target,
        change_weight=model.config.change_weight,
    )
    score = -(char_loss + model.config.color_loss_weight * color_loss)
    return score.reshape(batch_size, candidate_count)


def _per_example_categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    change_weight: float,
) -> torch.Tensor:
    raw = F.cross_entropy(logits, target, reduction="none")
    weights = torch.where(target == 0, 1.0, change_weight)
    weighted = raw * weights
    return weighted.flatten(1).sum(dim=1) / weights.flatten(1).sum(dim=1)


def _reveal_high_confidence(
    *,
    char_delta: torch.Tensor,
    color_delta: torch.Tensor,
    masked: torch.Tensor,
    char_prediction: torch.Tensor,
    color_prediction: torch.Tensor,
    confidence: torch.Tensor,
    remaining: int,
) -> None:
    for batch_index in range(masked.shape[0]):
        flat_mask = masked[batch_index].flatten()
        reveal_count = int(flat_mask.sum().item()) - remaining
        if reveal_count <= 0:
            continue
        masked_indices = flat_mask.nonzero(as_tuple=False).squeeze(1)
        masked_confidence = confidence[batch_index].flatten()[masked_indices]
        reveal_count = min(reveal_count, int(masked_indices.numel()))
        selected = masked_indices[torch.topk(masked_confidence, k=reveal_count).indices]
        char_delta[batch_index].view(-1)[selected] = char_prediction[
            batch_index
        ].flatten()[selected]
        color_delta[batch_index].view(-1)[selected] = color_prediction[
            batch_index
        ].flatten()[selected]
        masked[batch_index].view(-1)[selected] = False
