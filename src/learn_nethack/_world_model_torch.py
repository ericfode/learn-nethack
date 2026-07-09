"""PyTorch implementation for the optional local world-model proof."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from learn_nethack.world_model_metrics import (
    CHAR_CLASSES,
    CHAR_MASK_TOKEN,
    COLOR_CLASSES,
    COLOR_MASK_TOKEN,
    COLOR_VALUES,
)


MetricCallback = Callable[[dict[str, float | int], int], None]


@dataclass(frozen=True)
class TerminalWorldModelConfig:
    action_vocab_size: int = 121
    hidden_channels: int = 48
    residual_blocks: int = 4
    char_embedding_dim: int = 16
    color_embedding_dim: int = 4
    condition_dim: int = 48
    diffusion_steps: int = 6
    change_weight: float = 8.0
    color_loss_weight: float = 0.25


@dataclass(frozen=True)
class LocalTrainConfig:
    steps: int = 800
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 25
    validation_batches: int = 4
    seed: int = 20260709


class ConditionedResidualBlock(nn.Module):
    def __init__(self, hidden_channels: int, condition_dim: int) -> None:
        super().__init__()
        groups = _group_count(hidden_channels)
        self.norm1 = nn.GroupNorm(groups, hidden_channels)
        self.norm2 = nn.GroupNorm(groups, hidden_channels)
        self.conv1 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.condition = nn.Linear(condition_dim, hidden_channels * 2)

    def forward(self, inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.norm1(inputs)
        hidden = hidden * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.conv1(F.silu(hidden))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return inputs + hidden


class TerminalDeltaModel(nn.Module):
    """Action-conditioned full-terminal categorical delta predictor."""

    def __init__(self, config: TerminalWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.current_char_embedding = nn.Embedding(256, config.char_embedding_dim)
        self.current_color_embedding = nn.Embedding(
            COLOR_VALUES,
            config.color_embedding_dim,
        )
        self.delta_char_embedding = nn.Embedding(
            CHAR_CLASSES + 1,
            config.char_embedding_dim,
        )
        self.delta_color_embedding = nn.Embedding(
            COLOR_CLASSES + 1,
            config.color_embedding_dim,
        )
        input_channels = 2 * (config.char_embedding_dim + config.color_embedding_dim)
        self.input_projection = nn.Conv2d(
            input_channels,
            config.hidden_channels,
            1,
        )
        self.action_embedding = nn.Embedding(
            config.action_vocab_size,
            config.condition_dim,
        )
        self.time_embedding = nn.Embedding(
            config.diffusion_steps + 1,
            config.condition_dim,
        )
        self.blocks = nn.ModuleList(
            ConditionedResidualBlock(
                config.hidden_channels,
                config.condition_dim,
            )
            for _ in range(config.residual_blocks)
        )
        self.output_norm = nn.GroupNorm(
            _group_count(config.hidden_channels),
            config.hidden_channels,
        )
        self.char_head = nn.Conv2d(config.hidden_channels, CHAR_CLASSES, 1)
        self.color_head = nn.Conv2d(config.hidden_channels, COLOR_CLASSES, 1)

    def forward(
        self,
        current_chars: torch.Tensor,
        current_colors: torch.Tensor,
        noisy_char_delta: torch.Tensor,
        noisy_color_delta: torch.Tensor,
        action_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            (
                self.current_char_embedding(current_chars),
                self.current_color_embedding(current_colors),
                self.delta_char_embedding(noisy_char_delta),
                self.delta_color_embedding(noisy_color_delta),
            ),
            dim=-1,
        ).permute(0, 3, 1, 2)
        hidden = self.input_projection(features)
        condition = self.action_embedding(action_ids) + self.time_embedding(timesteps)
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = F.silu(self.output_norm(hidden))
        return self.char_head(hidden), self.color_head(hidden)


def build_model(config: TerminalWorldModelConfig, *, seed: int) -> TerminalDeltaModel:
    seed_everything(seed)
    return TerminalDeltaModel(config)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_variant(
    *,
    variant: str,
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    model_config: TerminalWorldModelConfig,
    train_config: LocalTrainConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    metric_callback: MetricCallback | None = None,
) -> dict[str, object]:
    if variant not in {"deterministic", "diffusion"}:
        raise ValueError(f"unsupported world-model variant: {variant}")
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("train and validation indices must not be empty")
    model = build_model(model_config, seed=train_config.seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    rng = np.random.default_rng(train_config.seed)
    started = time.perf_counter()
    recent_losses: list[float] = []
    last_metrics: dict[str, float | int] = {}
    model.train()
    for step in range(1, train_config.steps + 1):
        batch_indices = rng.choice(
            train_indices,
            size=train_config.batch_size,
            replace=train_indices.size < train_config.batch_size,
        )
        batch = tensor_batch(arrays, batch_indices, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, components = training_loss(
            model,
            batch,
            variant=variant,
            config=model_config,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            train_config.grad_clip,
        )
        optimizer.step()
        recent_losses.append(float(loss.detach().cpu()))

        if step == 1 or step % train_config.log_interval == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            validation_loss = evaluate_validation_loss(
                model=model,
                variant=variant,
                arrays=arrays,
                indices=validation_indices,
                model_config=model_config,
                train_config=train_config,
                device=device,
            )
            last_metrics = {
                "train/loss": float(np.mean(recent_losses)),
                "train/char_loss": components["char_loss"],
                "train/color_loss": components["color_loss"],
                "train/grad_norm": float(grad_norm.detach().cpu()),
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train/examples_per_second": (step * train_config.batch_size / elapsed),
                "validation/loss": validation_loss,
            }
            if metric_callback is not None:
                metric_callback(last_metrics, step)
            recent_losses.clear()

    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "learn-nethack.local-world-model-checkpoint.v1",
            "variant": variant,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "state_dict": model.state_dict(),
        },
        target,
    )
    return {
        "variant": variant,
        "checkpoint_path": str(target),
        "parameter_count": parameter_count(model),
        "training_seconds": time.perf_counter() - started,
        "final_metrics": last_metrics,
    }


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[TerminalDeltaModel, str, dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model_config = TerminalWorldModelConfig(**payload["model_config"])
    model = TerminalDeltaModel(model_config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, str(payload["variant"]), payload


def tensor_batch(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(arrays[name][indices], device=device, dtype=torch.long)
        for name in (
            "current_chars",
            "current_colors",
            "next_chars",
            "next_colors",
            "action_ids",
        )
    }


def training_loss(
    model: TerminalDeltaModel,
    batch: dict[str, torch.Tensor],
    *,
    variant: str,
    config: TerminalWorldModelConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    char_target, color_target = delta_targets(batch)
    batch_size, rows, cols = char_target.shape
    if variant == "deterministic":
        mask = torch.ones(
            (batch_size, rows, cols),
            dtype=torch.bool,
            device=char_target.device,
        )
        timesteps = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=char_target.device,
        )
    elif variant == "diffusion":
        timesteps = torch.randint(
            1,
            config.diffusion_steps + 1,
            (batch_size,),
            device=char_target.device,
        )
        probability = timesteps.float() / float(config.diffusion_steps)
        mask = (
            torch.rand(
                (batch_size, rows, cols),
                device=char_target.device,
            )
            < probability[:, None, None]
        )
        mask[:, 0, 0] = True
    else:
        raise ValueError(f"unsupported world-model variant: {variant}")
    noisy_char = torch.where(mask, CHAR_MASK_TOKEN, char_target)
    noisy_color = torch.where(mask, COLOR_MASK_TOKEN, color_target)
    char_logits, color_logits = model(
        batch["current_chars"],
        batch["current_colors"],
        noisy_char,
        noisy_color,
        batch["action_ids"],
        timesteps,
    )
    char_loss = _weighted_categorical_loss(
        char_logits,
        char_target,
        mask,
        change_weight=config.change_weight,
    )
    color_loss = _weighted_categorical_loss(
        color_logits,
        color_target,
        mask,
        change_weight=config.change_weight,
    )
    loss = char_loss + config.color_loss_weight * color_loss
    return loss, {
        "char_loss": float(char_loss.detach().cpu()),
        "color_loss": float(color_loss.detach().cpu()),
    }


@torch.inference_mode()
def evaluate_validation_loss(
    *,
    model: TerminalDeltaModel,
    variant: str,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    model_config: TerminalWorldModelConfig,
    train_config: LocalTrainConfig,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    count = min(
        indices.size,
        train_config.validation_batches * train_config.batch_size,
    )
    for offset in range(0, count, train_config.batch_size):
        selected = indices[offset : offset + train_config.batch_size]
        batch = tensor_batch(arrays, selected, device=device)
        loss, _ = training_loss(
            model,
            batch,
            variant=variant,
            config=model_config,
        )
        losses.append(float(loss.cpu()))
    if was_training:
        model.train()
    return float(np.mean(losses))


def delta_targets(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    char_target = torch.where(
        batch["current_chars"] == batch["next_chars"],
        0,
        batch["next_chars"] + 1,
    )
    color_target = torch.where(
        batch["current_colors"] == batch["next_colors"],
        0,
        batch["next_colors"] + 1,
    )
    return char_target, color_target


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _weighted_categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    *,
    change_weight: float,
) -> torch.Tensor:
    raw = F.cross_entropy(logits, target, reduction="none")
    weights = torch.where(target == 0, 1.0, change_weight)
    active_weights = weights * active
    return (raw * active_weights).sum() / active_weights.sum().clamp_min(1.0)


def _group_count(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1
