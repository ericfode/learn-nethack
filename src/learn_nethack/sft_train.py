"""Unsloth/TRL SFT training contracts for NetHack rows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import importlib
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from learn_nethack.wandb_logging import resolve_local_wandb_mode

TRAINING_OBJECTIVES = ("policy_dynamics_phased", "policy_only", "dynamics_only")


@dataclass(frozen=True)
class SftTrainConfig:
    model_name: str = "google/gemma-4-E4b-it"
    max_seq_length: int = 2048
    load_in_16bit: bool = True
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    max_steps: int = 100
    logging_steps: int = 1
    seed: int = 3407
    use_gradient_checkpointing: bool | str = False
    train_on_assistant_only: bool = True
    wandb_project: str = "learn-nethack"
    torch_dynamo_recompile_limit: int = 128
    dynamics_warmup_steps: int = 50
    mixed_training_steps: int = 100
    policy_calibration_steps: int = 20
    frame_auxiliary_ratio: float = 0.25
    frame_loss_weight: float = 0.25
    max_next_frame_chars: int = 4096
    training_objective: str = "policy_dynamics_phased"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class ResolvedJsonlTrainingConfig:
    config: SftTrainConfig
    max_steps_mode: str
    requested_max_steps: int
    train_row_count: int
    effective_batch_size: int


def require_wandb_for_training(env: Mapping[str, str | None] | None = None) -> str:
    """Resolve W&B mode and fail before GPU work if reporting is not configured."""
    return resolve_local_wandb_mode(env)


def configure_training_runtime(config: SftTrainConfig) -> None:
    """Set process-level trainer runtime defaults before model construction."""
    os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
    torch = _try_import_module("torch")
    dynamo = getattr(torch, "_dynamo", None) if torch is not None else None
    dynamo_config = getattr(dynamo, "config", None)
    if dynamo_config is None:
        return
    for attr in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit"):
        if hasattr(dynamo_config, attr):
            current = getattr(dynamo_config, attr)
            if isinstance(current, int):
                setattr(
                    dynamo_config,
                    attr,
                    max(current, config.torch_dynamo_recompile_limit),
                )


def _try_import_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def apply_gradient_checkpointing_contract(model: Any, config: SftTrainConfig) -> None:
    """Make model checkpointing match the explicit SFT config."""
    if config.use_gradient_checkpointing:
        return
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "gradient_checkpointing"):
        setattr(model_config, "gradient_checkpointing", False)


def build_phase_rows(
    rows: Sequence[dict[str, Any]],
    config: SftTrainConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Split multitask rows into the three SFT phases."""
    objective = _validate_training_objective(config.training_objective)
    policy_rows = [row for row in rows if row.get("task") == "policy_action"]
    next_frame_rows = [row for row in rows if row.get("task") == "next_frame"]
    if objective == "policy_only":
        return {"policy_only": list(policy_rows)}
    if objective == "dynamics_only":
        return {"dynamics_only": list(next_frame_rows)}
    sampled_next_frame = _sample_rows(
        next_frame_rows,
        ratio=config.frame_auxiliary_ratio,
        seed=config.seed,
    )
    return {
        "dynamics_warmup": list(next_frame_rows),
        "mixed": list(policy_rows) + sampled_next_frame,
        "policy_calibration": list(policy_rows),
    }


def _sample_rows(
    rows: Sequence[dict[str, Any]], *, ratio: float, seed: int
) -> list[dict[str, Any]]:
    if ratio <= 0 or not rows:
        return []
    count = min(len(rows), int(len(rows) * ratio))
    if count <= 0:
        return []
    indexed_rows = list(enumerate(rows))
    selected = random.Random(seed).sample(indexed_rows, count)
    return [row for _index, row in sorted(selected, key=lambda item: item[0])]


def format_row_for_sft(row: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Render a chat row for text-field SFT consumers while preserving task labels."""
    text = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {
        "text": text,
        "task": row.get("task"),
    }


def tokenize_row_for_assistant_only_loss(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, Any]:
    """Tokenize one chat row and mask everything before the final response."""
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("SFT row must contain at least two chat messages")
    if not all(isinstance(message, Mapping) for message in messages):
        raise ValueError("SFT row messages must be objects")
    if messages[-1].get("role") != "assistant":
        raise ValueError("SFT row must end with an assistant message")

    prompt_ids = _chat_template_token_ids(
        tokenizer,
        messages[:-1],
        add_generation_prompt=True,
    )
    full_ids = _chat_template_token_ids(
        tokenizer,
        messages,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template does not produce a verifiable final assistant suffix"
        )
    assistant_token_count = len(full_ids) - len(prompt_ids)
    if assistant_token_count <= 0:
        raise ValueError("chat template produced no final assistant tokens")

    input_ids = full_ids[:max_seq_length]
    masked_prompt_tokens = min(len(prompt_ids), len(input_ids))
    supervised_tokens = len(input_ids) - masked_prompt_tokens
    if supervised_tokens <= 0:
        raise ValueError(
            "max_seq_length truncates every final assistant token; shorten the prompt "
            "or increase max_seq_length"
        )
    labels = [-100] * masked_prompt_tokens + input_ids[masked_prompt_tokens:]
    assistant_tokens_truncated = max(0, assistant_token_count - supervised_tokens)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "task": row.get("task"),
        "input_token_count": len(input_ids),
        "prompt_token_count": masked_prompt_tokens,
        "masked_prompt_token_count": masked_prompt_tokens,
        "assistant_token_count": assistant_token_count,
        "supervised_assistant_token_count": supervised_tokens,
        "assistant_tokens_truncated": assistant_tokens_truncated,
        "truncated_token_count": max(0, len(full_ids) - len(input_ids)),
    }


def build_assistant_mask_report(
    tokenized_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate auditable assistant-only token counts for a trainer dataset."""
    totals = _empty_assistant_mask_counts()
    per_task: dict[str, dict[str, int]] = {}
    for row in tokenized_rows:
        task = str(row.get("task") or "unknown")
        task_counts = per_task.setdefault(task, _empty_assistant_mask_counts())
        for counts in (totals, task_counts):
            counts["row_count"] += 1
            for key in (
                "input_token_count",
                "prompt_token_count",
                "masked_prompt_token_count",
                "assistant_token_count",
                "supervised_assistant_token_count",
                "assistant_tokens_truncated",
                "truncated_token_count",
            ):
                counts[key] += int(row[key])
            if int(row["assistant_tokens_truncated"]) > 0:
                counts["rows_with_truncated_assistant"] += 1

    supervised = totals["supervised_assistant_token_count"]
    input_tokens = totals["input_token_count"]
    report: dict[str, Any] = {
        "schema_version": "learn-nethack.assistant-mask-report.v1",
        **totals,
        "supervised_token_fraction": (
            supervised / input_tokens if input_tokens else 0.0
        ),
        "per_task": {},
    }
    for task, counts in sorted(per_task.items()):
        task_input_tokens = counts["input_token_count"]
        report["per_task"][task] = {
            **counts,
            "supervised_token_fraction": (
                counts["supervised_assistant_token_count"] / task_input_tokens
                if task_input_tokens
                else 0.0
            ),
        }
    if totals["row_count"] <= 0:
        raise ValueError("assistant-mask report requires at least one row")
    if supervised <= 0:
        raise ValueError("assistant-mask report contains no supervised tokens")
    return report


def get_trainer_assistant_mask_report(trainer: Any) -> dict[str, Any]:
    """Return the verified assistant-only mask report attached to a trainer."""
    report = getattr(trainer, "learn_nethack_assistant_mask_report", None)
    if not isinstance(report, dict):
        raise RuntimeError("SFT trainer is missing its assistant-mask report")
    return report


def summarize_trainer_loss_history(
    history: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize logged training losses for a small-run learning gate."""
    losses = [
        float(entry["loss"])
        for entry in history
        if isinstance(entry.get("loss"), int | float)
    ]
    return {
        "logged_loss_count": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "minimum_loss": min(losses) if losses else None,
        "loss_decreased": len(losses) >= 2 and losses[-1] < losses[0],
    }


def _chat_template_token_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    tolist = getattr(encoded, "tolist", None)
    if callable(tolist):
        encoded = tolist()
    if isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in encoded
    ):
        raise ValueError("chat template must return a flat list of integer token IDs")
    return encoded


def _empty_assistant_mask_counts() -> dict[str, int]:
    return {
        "row_count": 0,
        "input_token_count": 0,
        "prompt_token_count": 0,
        "masked_prompt_token_count": 0,
        "assistant_token_count": 0,
        "supervised_assistant_token_count": 0,
        "assistant_tokens_truncated": 0,
        "truncated_token_count": 0,
        "rows_with_truncated_assistant": 0,
    }


def load_jsonl_rows(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load SFT JSONL rows from one or more local artifact files."""
    import json

    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def create_unsloth_sft_trainer(
    *,
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    config: SftTrainConfig,
    env: Mapping[str, str | None] | None = None,
):
    """Create a TRL SFTTrainer backed by an Unsloth LoRA model."""
    require_wandb_for_training(env)
    configure_training_runtime(config)

    try:
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - requires GPU training deps.
        raise RuntimeError(
            "datasets, trl, and unsloth are required to create the SFT trainer"
        ) from exc
    configure_training_runtime(config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_16bit=config.load_in_16bit,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=list(config.target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
    )
    apply_gradient_checkpointing_contract(model, config)
    if not config.train_on_assistant_only:
        raise ValueError("NetHack SFT requires train_on_assistant_only=True")
    tokenized_rows = [
        tokenize_row_for_assistant_only_loss(
            row,
            tokenizer,
            max_seq_length=config.max_seq_length,
        )
        for row in rows
    ]
    assistant_mask_report = build_assistant_mask_report(tokenized_rows)
    dataset = Dataset.from_list(tokenized_rows)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=config.max_seq_length,
        completion_only_loss=False,
        assistant_only_loss=False,
        gradient_checkpointing=bool(config.use_gradient_checkpointing),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        report_to="wandb",
        run_name=Path(output_dir).name,
        seed=config.seed,
        dataset_num_proc=None,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.learn_nethack_assistant_mask_report = assistant_mask_report
    return trainer


def build_sft_jsonl_training_plan(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    config: SftTrainConfig,
) -> dict[str, Any]:
    """Return the full-dataset JSONL training plan without loading examples."""
    train_path = Path(dataset_dir) / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"required SFT train file is missing: {train_path}")
    tasks = _manifest_tasks(Path(dataset_dir) / "manifest.json")
    return {
        "schema_version": "learn-nethack.sft-jsonl-train-plan.v1",
        "train_files": [str(train_path)],
        "output_dir": str(Path(output_dir)),
        "model_name": config.model_name,
        "max_seq_length": config.max_seq_length,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "tasks": tasks,
    }


def build_sft_jsonl_curriculum_plan(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    scratch_dir: str | Path,
    config: SftTrainConfig,
    requested_max_steps: int,
) -> dict[str, Any]:
    """Return the phased JSONL training plan used by existing-dataset SFT."""
    dataset_root = Path(dataset_dir)
    combined_plan = build_sft_jsonl_training_plan(
        dataset_dir=dataset_root,
        output_dir=output_dir,
        config=config,
    )
    tasks = set(combined_plan["tasks"])
    policy_path = dataset_root / "train.policy_action.jsonl"
    next_frame_path = dataset_root / "train.next_frame.jsonl"
    effective_batch_size = (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    if effective_batch_size <= 0:
        raise ValueError("effective training batch size must be positive")
    objective = _validate_training_objective(config.training_objective)
    if objective == "policy_only":
        return _single_task_curriculum_plan(
            combined_plan=combined_plan,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=config,
            requested_max_steps=requested_max_steps,
            effective_batch_size=effective_batch_size,
            task="policy_action",
            phase_name="policy_only",
        )
    if objective == "dynamics_only":
        return _single_task_curriculum_plan(
            combined_plan=combined_plan,
            dataset_root=dataset_root,
            output_dir=output_dir,
            config=config,
            requested_max_steps=requested_max_steps,
            effective_batch_size=effective_batch_size,
            task="next_frame",
            phase_name="dynamics_only",
        )
    if not {"policy_action", "next_frame"}.issubset(tasks) or not (
        policy_path.exists() and next_frame_path.exists()
    ):
        return {
            **combined_plan,
            "schema_version": "learn-nethack.sft-jsonl-curriculum-plan.v1",
            "curriculum": "single_phase",
            "requested_max_steps": requested_max_steps,
            "total_max_steps": config.max_steps,
            "effective_batch_size": effective_batch_size,
            "phases": [
                {
                    "name": "combined",
                    "train_files": combined_plan["train_files"],
                    "max_steps": config.max_steps,
                    "row_count": _count_jsonl_rows(dataset_root / "train.jsonl"),
                    "tasks": combined_plan["tasks"],
                }
            ],
        }

    policy_count = _count_jsonl_rows(policy_path)
    next_frame_count = _count_jsonl_rows(next_frame_path)
    if policy_count <= 0:
        raise ValueError(f"policy training file has no rows: {policy_path}")
    if next_frame_count <= 0:
        raise ValueError(f"next-frame training file has no rows: {next_frame_path}")

    scratch_root = Path(scratch_dir)
    scratch_root.mkdir(parents=True, exist_ok=True)
    mixed_path = scratch_root / "train.mixed.curriculum.jsonl"
    policy_mixed_count, sampled_next_frame_count = write_mixed_curriculum_jsonl(
        policy_path=policy_path,
        next_frame_path=next_frame_path,
        output_path=mixed_path,
        ratio=config.frame_auxiliary_ratio,
        seed=config.seed,
    )
    mixed_row_count = policy_mixed_count + sampled_next_frame_count
    if requested_max_steps <= 0:
        warmup_steps = min(
            config.dynamics_warmup_steps,
            max(1, math.ceil(next_frame_count / effective_batch_size)),
        )
        mixed_steps = max(1, math.ceil(mixed_row_count / effective_batch_size))
        calibration_steps = min(
            config.policy_calibration_steps,
            max(1, math.ceil(policy_count / effective_batch_size)),
        )
        max_steps_mode = "auto_curriculum_one_mixed_pass"
    else:
        warmup_steps, mixed_steps, calibration_steps = _split_explicit_phase_steps(
            total_steps=config.max_steps,
            config=config,
        )
        max_steps_mode = "explicit_curriculum"

    phases = [
        {
            "name": "dynamics_warmup",
            "train_files": [str(next_frame_path)],
            "max_steps": warmup_steps,
            "row_count": next_frame_count,
            "tasks": ["next_frame"],
        },
        {
            "name": "mixed",
            "train_files": [str(mixed_path)],
            "max_steps": mixed_steps,
            "row_count": mixed_row_count,
            "tasks": ["policy_action", "next_frame"]
            if sampled_next_frame_count > 0
            else ["policy_action"],
            "sampled_next_frame_rows": sampled_next_frame_count,
            "frame_auxiliary_ratio": config.frame_auxiliary_ratio,
        },
        {
            "name": "policy_calibration",
            "train_files": [str(policy_path)],
            "max_steps": calibration_steps,
            "row_count": policy_count,
            "tasks": ["policy_action"],
        },
    ]
    phases = [phase for phase in phases if int(phase["max_steps"]) > 0]
    return {
        **combined_plan,
        "schema_version": "learn-nethack.sft-jsonl-curriculum-plan.v1",
        "curriculum": "policy_dynamics_phased",
        "training_objective": objective,
        "requested_max_steps": requested_max_steps,
        "max_steps_mode": max_steps_mode,
        "total_max_steps": sum(int(phase["max_steps"]) for phase in phases),
        "effective_batch_size": effective_batch_size,
        "policy_row_count": policy_count,
        "next_frame_row_count": next_frame_count,
        "sampled_next_frame_rows": sampled_next_frame_count,
        "frame_auxiliary_ratio": config.frame_auxiliary_ratio,
        "frame_loss_weight": config.frame_loss_weight,
        "phases": phases,
    }


def _single_task_curriculum_plan(
    *,
    combined_plan: dict[str, Any],
    dataset_root: Path,
    output_dir: str | Path,
    config: SftTrainConfig,
    requested_max_steps: int,
    effective_batch_size: int,
    task: str,
    phase_name: str,
) -> dict[str, Any]:
    task_path = dataset_root / f"train.{task}.jsonl"
    if not task_path.exists():
        task_path = dataset_root / "train.jsonl"
    row_count = _count_jsonl_rows(task_path)
    if row_count <= 0:
        raise ValueError(f"SFT {task} train file has no rows: {task_path}")
    if requested_max_steps <= 0:
        phase_steps = max(1, math.ceil(row_count / effective_batch_size))
        max_steps_mode = f"auto_{phase_name}_one_pass"
    else:
        phase_steps = config.max_steps
        max_steps_mode = f"explicit_{phase_name}"
    return {
        **combined_plan,
        "schema_version": "learn-nethack.sft-jsonl-curriculum-plan.v1",
        "curriculum": phase_name,
        "training_objective": config.training_objective,
        "requested_max_steps": requested_max_steps,
        "max_steps_mode": max_steps_mode,
        "total_max_steps": phase_steps,
        "effective_batch_size": effective_batch_size,
        "policy_row_count": row_count if task == "policy_action" else 0,
        "next_frame_row_count": row_count if task == "next_frame" else 0,
        "sampled_next_frame_rows": 0,
        "frame_auxiliary_ratio": 0.0 if task == "policy_action" else 1.0,
        "frame_loss_weight": config.frame_loss_weight,
        "output_dir": str(Path(output_dir)),
        "phases": [
            {
                "name": phase_name,
                "train_files": [str(task_path)],
                "max_steps": phase_steps,
                "row_count": row_count,
                "tasks": [task],
            }
        ],
    }


def _split_explicit_phase_steps(
    *, total_steps: int, config: SftTrainConfig
) -> tuple[int, int, int]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if total_steps == 1:
        return 0, 1, 0
    warmup_steps = min(config.dynamics_warmup_steps, max(1, total_steps // 5))
    calibration_steps = min(
        config.policy_calibration_steps,
        max(1, total_steps // 10),
        max(0, total_steps - warmup_steps - 1),
    )
    mixed_steps = total_steps - warmup_steps - calibration_steps
    if mixed_steps <= 0:
        mixed_steps = 1
        if calibration_steps > 0:
            calibration_steps -= 1
        else:
            warmup_steps = max(0, warmup_steps - 1)
    return warmup_steps, mixed_steps, calibration_steps


def write_mixed_curriculum_jsonl(
    *,
    policy_path: str | Path,
    next_frame_path: str | Path,
    output_path: str | Path,
    ratio: float,
    seed: int,
) -> tuple[int, int]:
    """Write policy rows plus sampled frame rows with one minimal JSONL schema."""
    target = Path(output_path)
    policy_count = 0
    frame_count = 0
    with target.open("w", encoding="utf-8") as target_handle:
        with Path(policy_path).open(encoding="utf-8") as policy_handle:
            for line in policy_handle:
                if not line.strip():
                    continue
                row = _minimal_sft_row(line=line, source_path=Path(policy_path))
                target_handle.write(json.dumps(row, sort_keys=True) + "\n")
                policy_count += 1
        if ratio > 0:
            threshold = 1_000_000 if ratio >= 1 else int(ratio * 1_000_000)
            with Path(next_frame_path).open(encoding="utf-8") as frame_handle:
                for index, line in enumerate(frame_handle):
                    if not line.strip():
                        continue
                    draw = (
                        0
                        if ratio >= 1
                        else random.Random(seed + index).randrange(1_000_000)
                    )
                    if draw < threshold:
                        row = _minimal_sft_row(
                            line=line,
                            source_path=Path(next_frame_path),
                        )
                        target_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        frame_count += 1
    return policy_count, frame_count


def _minimal_sft_row(*, line: str, source_path: Path) -> dict[str, Any]:
    payload = json.loads(line)
    messages = payload.get("messages")
    task = payload.get("task")
    if not isinstance(messages, list):
        raise ValueError(f"SFT row is missing messages: {source_path}")
    if not isinstance(task, str):
        raise ValueError(f"SFT row is missing task: {source_path}")
    return {"messages": messages, "task": task}


def write_sampled_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    ratio: float,
    seed: int,
) -> int:
    """Write a deterministic sampled JSONL subset and return rows written."""
    source = Path(input_path)
    target = Path(output_path)
    if ratio <= 0:
        target.write_text("", encoding="utf-8")
        return 0
    if ratio >= 1:
        written = 0
        with source.open(encoding="utf-8") as source_handle:
            with target.open("w", encoding="utf-8") as target_handle:
                for line in source_handle:
                    if line.strip():
                        target_handle.write(line)
                        written += 1
        return written
    threshold = int(ratio * 1_000_000)
    written = 0
    with source.open(encoding="utf-8") as source_handle:
        with target.open("w", encoding="utf-8") as target_handle:
            for index, line in enumerate(source_handle):
                if not line.strip():
                    continue
                draw = random.Random(seed + index).randrange(1_000_000)
                if draw < threshold:
                    target_handle.write(line)
                    written += 1
    return written


def resolve_jsonl_training_config(
    *,
    dataset_dir: str | Path,
    config: SftTrainConfig,
) -> ResolvedJsonlTrainingConfig:
    """Resolve explicit or auto full-pass max_steps for existing JSONL training."""
    train_path = _training_rows_path_for_objective(
        dataset_dir=Path(dataset_dir),
        objective=config.training_objective,
    )
    train_row_count = _count_jsonl_rows(train_path)
    effective_batch_size = (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    if effective_batch_size <= 0:
        raise ValueError("effective training batch size must be positive")
    if train_row_count <= 0:
        raise ValueError(f"SFT train file has no rows: {train_path}")
    if config.max_steps > 0:
        return ResolvedJsonlTrainingConfig(
            config=config,
            max_steps_mode="explicit",
            requested_max_steps=config.max_steps,
            train_row_count=train_row_count,
            effective_batch_size=effective_batch_size,
        )
    resolved_steps = max(1, math.ceil(train_row_count / effective_batch_size))
    return ResolvedJsonlTrainingConfig(
        config=replace(config, max_steps=resolved_steps),
        max_steps_mode="auto_one_full_pass",
        requested_max_steps=config.max_steps,
        train_row_count=train_row_count,
        effective_batch_size=effective_batch_size,
    )


def _training_rows_path_for_objective(*, dataset_dir: Path, objective: str) -> Path:
    objective = _validate_training_objective(objective)
    if objective == "policy_only":
        return dataset_dir / "train.policy_action.jsonl"
    if objective == "dynamics_only":
        return dataset_dir / "train.next_frame.jsonl"
    return dataset_dir / "train.jsonl"


def _validate_training_objective(objective: str) -> str:
    if objective not in TRAINING_OBJECTIVES:
        allowed = ", ".join(TRAINING_OBJECTIVES)
        raise ValueError(
            f"unknown SFT training objective {objective!r}; expected {allowed}"
        )
    return objective


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"required SFT train file is missing: {path}")
    row_count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row_count += 1
    return row_count


def _manifest_tasks(manifest_path: Path) -> list[str]:
    if not manifest_path.exists():
        return ["policy_action", "next_frame"]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ValueError(f"SFT manifest has invalid tasks field: {manifest_path}")
    return tasks


def create_unsloth_sft_trainer_from_jsonl(
    *,
    jsonl_paths: Sequence[str | Path],
    output_dir: str | Path,
    config: SftTrainConfig,
    env: Mapping[str, str | None] | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
):
    """Create an Unsloth SFT trainer from JSONL files without materializing rows."""
    require_wandb_for_training(env)
    configure_training_runtime(config)
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together")
    paths = [Path(path) for path in jsonl_paths]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SFT JSONL files do not exist: {missing}")

    if model is None or tokenizer is None:
        model, tokenizer = load_unsloth_lora_model(config)
    try:
        from datasets import load_dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - requires GPU training deps.
        raise RuntimeError(
            "datasets and trl are required to create the SFT trainer"
        ) from exc
    configure_training_runtime(config)
    if not config.train_on_assistant_only:
        raise ValueError("NetHack SFT requires train_on_assistant_only=True")
    dataset = load_dataset(
        "json",
        data_files=[str(path) for path in paths],
        split="train",
    )

    tokenized_rows_for_report: list[dict[str, Any]] = []

    def _format_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        tasks = batch.get("task", [None] * len(batch["messages"]))
        tokenized_rows = []
        for messages, task in zip(batch["messages"], tasks):
            tokenized = tokenize_row_for_assistant_only_loss(
                {"messages": messages, "task": task},
                tokenizer,
                max_seq_length=config.max_seq_length,
            )
            tokenized_rows.append(tokenized)
            tokenized_rows_for_report.append(tokenized)
        return {key: [row[key] for row in tokenized_rows] for key in tokenized_rows[0]}

    dataset = dataset.map(
        _format_batch,
        batched=True,
        remove_columns=list(dataset.column_names),
        load_from_cache_file=False,
    )
    assistant_mask_report = build_assistant_mask_report(tokenized_rows_for_report)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=config.max_seq_length,
        completion_only_loss=False,
        assistant_only_loss=False,
        gradient_checkpointing=bool(config.use_gradient_checkpointing),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        report_to="wandb",
        run_name=Path(output_dir).name,
        seed=config.seed,
        dataset_num_proc=None,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.learn_nethack_assistant_mask_report = assistant_mask_report
    return trainer


def load_unsloth_lora_model(config: SftTrainConfig) -> tuple[Any, Any]:
    """Load the base model once and attach the trainable LoRA adapter."""
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - requires GPU training deps.
        raise RuntimeError("unsloth is required to load the SFT model") from exc
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        load_in_16bit=config.load_in_16bit,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=list(config.target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.seed,
    )
    apply_gradient_checkpointing_contract(model, config)
    return model, tokenizer
