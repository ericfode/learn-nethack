"""CLI and orchestration for the local diffusion-decoder falsification run."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
from typing import Any

import typer

from learn_nethack.action_manifest import load_action_manifest
from learn_nethack.wandb_logging import resolve_local_wandb_mode
from learn_nethack.world_model_data import (
    LocalWorldModelDataConfig,
    build_local_world_model_dataset,
    load_transition_arrays,
)
from learn_nethack.world_model_aggregate import aggregate_world_model_reports
from learn_nethack.world_model_eval import (
    WorldModelEvalConfig,
    build_rollout_starts,
    evaluate_copy_baseline,
    evaluate_checkpoint,
    write_comparison_watch,
)
from learn_nethack.world_model_metrics import paired_bootstrap_interval
from learn_nethack.world_model_wandb import (
    flatten_numeric_report,
    log_wandb_artifact,
    progress_callback,
    start_wandb_run,
    wandb_identity,
)


app = typer.Typer(help="Run the local NetHack terminal world-model proof.")


@app.command("build-data")
def build_data(
    db: Path = typer.Option(..., "--db"),
    action_manifest: Path = typer.Option(..., "--action-manifest"),
    out: Path = typer.Option(..., "--out"),
    seed: int = typer.Option(20260709, "--seed"),
    train_games: int = typer.Option(48, "--train-games"),
    validation_games: int = typer.Option(12, "--validation-games"),
    test_games: int = typer.Option(12, "--test-games"),
    train_transitions: int = typer.Option(12_000, "--train-transitions"),
    validation_transitions: int = typer.Option(2_000, "--validation-transitions"),
    test_transitions: int = typer.Option(2_000, "--test-transitions"),
    decoder_workers: int = typer.Option(12, "--decoder-workers"),
) -> None:
    """Build compact true-keypress terminal transitions from the NLD-AA source."""
    result = build_local_world_model_dataset(
        db_path=db,
        action_manifest_path=action_manifest,
        out_dir=out,
        config=LocalWorldModelDataConfig(
            seed=seed,
            train_games=train_games,
            validation_games=validation_games,
            test_games=test_games,
            train_transitions=train_transitions,
            validation_transitions=validation_transitions,
            test_transitions=test_transitions,
            decoder_workers=decoder_workers,
        ),
    )
    typer.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


@app.command("run")
def run(
    run_id: str = typer.Option(..., "--run-id"),
    dataset: Path = typer.Option(..., "--dataset"),
    dataset_manifest: Path = typer.Option(..., "--dataset-manifest"),
    action_manifest: Path = typer.Option(..., "--action-manifest"),
    out: Path = typer.Option(..., "--out"),
    steps: int = typer.Option(800, "--steps"),
    batch_size: int = typer.Option(16, "--batch-size"),
    learning_rate: float = typer.Option(3e-4, "--learning-rate"),
    hidden_channels: int = typer.Option(48, "--hidden-channels"),
    residual_blocks: int = typer.Option(4, "--residual-blocks"),
    diffusion_steps: int = typer.Option(6, "--diffusion-steps"),
    one_step_examples: int = typer.Option(256, "--one-step-examples"),
    rollout_examples: int = typer.Option(128, "--rollout-examples"),
    action_ranking_examples: int = typer.Option(256, "--action-ranking-examples"),
    seed: int = typer.Option(20260709, "--seed"),
    eval_seed: int | None = typer.Option(None, "--eval-seed"),
    device: str | None = typer.Option(None, "--device"),
    wandb_project: str = typer.Option("learn-nethack", "--wandb-project"),
) -> None:
    """Train both matched arms and write the pre-registered verdict."""
    report = run_local_world_model_proof(
        run_id=run_id,
        dataset_path=dataset,
        dataset_manifest_path=dataset_manifest,
        action_manifest_path=action_manifest,
        out_dir=out,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_channels=hidden_channels,
        residual_blocks=residual_blocks,
        diffusion_steps=diffusion_steps,
        one_step_examples=one_step_examples,
        rollout_examples=rollout_examples,
        action_ranking_examples=action_ranking_examples,
        seed=seed,
        eval_seed=eval_seed,
        requested_device=device,
        wandb_project=wandb_project,
    )
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@app.command("aggregate")
def aggregate(
    report: list[Path] = typer.Option(..., "--report"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """Aggregate matched proof reports across training seeds."""
    result = aggregate_world_model_reports(report_paths=report, out_path=out)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


def run_local_world_model_proof(
    *,
    run_id: str,
    dataset_path: str | Path,
    dataset_manifest_path: str | Path,
    action_manifest_path: str | Path,
    out_dir: str | Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_channels: int,
    residual_blocks: int,
    diffusion_steps: int,
    one_step_examples: int,
    rollout_examples: int,
    action_ranking_examples: int,
    seed: int,
    eval_seed: int | None,
    requested_device: str | None,
    wandb_project: str,
) -> dict[str, Any]:
    import numpy as np

    from learn_nethack._world_model_torch import (
        LocalTrainConfig,
        TerminalWorldModelConfig,
        build_model,
        parameter_count,
        resolve_device,
        train_variant,
    )

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    mode = resolve_local_wandb_mode()
    device = resolve_device(requested_device)
    arrays = load_transition_arrays(dataset_path)
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(encoding="utf-8")
    )
    split_indices = {
        "train": np.flatnonzero(arrays["split_codes"] == 0),
        "validation": np.flatnonzero(arrays["split_codes"] == 1),
        "test": np.flatnonzero(arrays["split_codes"] == 2),
    }
    _validate_split_arrays(arrays, split_indices)
    action_manifest = load_action_manifest(action_manifest_path)
    action_vocab_size = max(action_manifest.valid_action_ids()) + 1
    model_config = TerminalWorldModelConfig(
        action_vocab_size=action_vocab_size,
        hidden_channels=hidden_channels,
        residual_blocks=residual_blocks,
        diffusion_steps=diffusion_steps,
    )
    train_config = LocalTrainConfig(
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    eval_config = WorldModelEvalConfig(
        batch_size=batch_size,
        one_step_examples=one_step_examples,
        rollout_examples=rollout_examples,
        action_ranking_examples=action_ranking_examples,
        seed=seed if eval_seed is None else eval_seed,
    )
    parity_models = [build_model(model_config, seed=seed) for _ in range(2)]
    parameter_counts = [parameter_count(model) for model in parity_models]
    if parameter_counts[0] != parameter_counts[1]:
        raise RuntimeError("matched decoder parameter counts differ")
    del parity_models

    contract = {
        "schema_version": "learn-nethack.local-world-model-contract.v1",
        "run_id": run_id,
        "goal": (
            "falsify whether an absorbing-mask categorical diffusion decoder "
            "improves true-action terminal transition modeling over a "
            "capacity-matched deterministic decoder"
        ),
        "dataset_path": str(dataset_path),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_sha256": dataset_manifest.get("dataset_sha256"),
        "action_manifest_path": str(action_manifest_path),
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "eval_config": asdict(eval_config),
        "matched_parameter_count": parameter_counts[0],
        "split_counts": {
            name: int(indices.size) for name, indices in split_indices.items()
        },
        "split_gameids": {
            name: sorted(set(int(value) for value in arrays["gameids"][indices]))
            for name, indices in split_indices.items()
        },
        "execution": {
            "backend": "local",
            "device": str(device),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "wandb_mode": mode,
        },
        "decision_rule": {
            "next_10_changed_f1_delta_positive": True,
            "next_10_paired_bootstrap_lower_above_zero": True,
            "action_ranking_mrr_delta_positive": True,
            "diffusion_action_ranking_above_random": True,
            "one_step_char_accuracy_delta_minimum": -0.002,
            "both_predict_changed_cells": True,
        },
    }
    _write_json(target / "run_contract.json", contract)
    progress_path = target / "training_progress.jsonl"
    wandb_run = None
    started = time.perf_counter()
    try:
        wandb_run = start_wandb_run(
            run_id=run_id,
            project=wandb_project,
            mode=mode,
            target=target,
            config=contract,
        )
        wandb_run_identity = wandb_identity(wandb_run, mode=mode)
        _write_json(target / "wandb_run.json", wandb_run_identity)
        wandb_run.log(
            {
                f"data/{name}_transitions": int(indices.size)
                for name, indices in split_indices.items()
            },
            step=0,
        )

        training: dict[str, Any] = {}
        for variant_index, variant in enumerate(("deterministic", "diffusion")):
            callback = progress_callback(
                variant=variant,
                progress_path=progress_path,
                wandb_run=wandb_run,
                step_offset=variant_index * steps,
            )
            training[variant] = train_variant(
                variant=variant,
                arrays=arrays,
                train_indices=split_indices["train"],
                validation_indices=split_indices["validation"],
                model_config=model_config,
                train_config=train_config,
                checkpoint_path=target / "checkpoints" / f"{variant}.pt",
                device=device,
                metric_callback=callback,
            )

        starts = build_rollout_starts(
            arrays,
            split_code=2,
            config=eval_config,
        )
        if starts[10].size < 64:
            raise RuntimeError(
                f"next-10 evaluation has only {starts[10].size} contiguous starts"
            )
        evaluations: dict[str, Any] = {
            "copy_current": evaluate_copy_baseline(
                arrays=arrays,
                test_indices=split_indices["test"],
                rollout_starts=starts,
                config=eval_config,
            )
        }
        outputs: dict[str, Any] = {}
        for variant in ("deterministic", "diffusion"):
            evaluations[variant], outputs[variant] = evaluate_checkpoint(
                checkpoint_path=target / "checkpoints" / f"{variant}.pt",
                arrays=arrays,
                test_indices=split_indices["test"],
                train_indices=split_indices["train"],
                rollout_starts=starts,
                config=eval_config,
                device=device,
            )
        comparison = _build_comparison(
            deterministic=evaluations["deterministic"],
            diffusion=evaluations["diffusion"],
            deterministic_next_10=outputs["deterministic"][10]["changed_f1"],
            diffusion_next_10=outputs["diffusion"][10]["changed_f1"],
            seed=seed,
        )
        watch = write_comparison_watch(
            out_dir=target / "watch",
            run_id=run_id,
            arrays=arrays,
            starts=starts[10],
            horizon=10,
            deterministic_outputs=outputs["deterministic"][10],
            diffusion_outputs=outputs["diffusion"][10],
            action_manifest_path=action_manifest_path,
        )
        report = {
            "schema_version": "learn-nethack.local-world-model-proof.v1",
            "run_id": run_id,
            "verdict": comparison["verdict"],
            "contract_path": str(target / "run_contract.json"),
            "wandb": wandb_run_identity,
            "training": training,
            "evaluation": evaluations,
            "comparison": comparison,
            "watch": watch,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(target / "report.json", report)
        np.savez_compressed(
            target / "paired_next_10_changed_f1.npz",
            deterministic=outputs["deterministic"][10]["changed_f1"],
            diffusion=outputs["diffusion"][10]["changed_f1"],
            starts=starts[10],
        )
        wandb_run.log(flatten_numeric_report(report), step=2 * steps + 1)
        log_wandb_artifact(
            wandb_run=wandb_run,
            run_id=run_id,
            target=target,
            dataset_manifest_path=dataset_manifest_path,
        )
        wandb_run.finish()
        return report
    except Exception as exc:
        failure = {
            "schema_version": "learn-nethack.local-world-model-failure.v1",
            "run_id": run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(target / "failure.json", failure)
        if wandb_run is not None:
            wandb_run.log({"run/failed": 1})
            wandb_run.finish(exit_code=1)
        raise


def _build_comparison(
    *,
    deterministic: dict[str, Any],
    diffusion: dict[str, Any],
    deterministic_next_10,
    diffusion_next_10,
    seed: int,
) -> dict[str, Any]:
    next_10_delta = (
        diffusion["rollouts"]["next_10"]["changed_cell_f1"]
        - deterministic["rollouts"]["next_10"]["changed_cell_f1"]
    )
    action_mrr_delta = (
        diffusion["action_ranking"]["mean_reciprocal_rank"]
        - deterministic["action_ranking"]["mean_reciprocal_rank"]
    )
    diffusion_action_mrr = diffusion["action_ranking"]["mean_reciprocal_rank"]
    random_action_mrr = diffusion["action_ranking"]["random_mean_reciprocal_rank"]
    one_step_char_delta = (
        diffusion["one_step"]["full_frame_char_accuracy"]
        - deterministic["one_step"]["full_frame_char_accuracy"]
    )
    interval = paired_bootstrap_interval(
        deterministic_next_10,
        diffusion_next_10,
        seed=seed,
    )
    criteria = {
        "next_10_changed_f1_delta_positive": next_10_delta > 0.0,
        "next_10_paired_bootstrap_lower_above_zero": interval["lower"] > 0.0,
        "action_ranking_mrr_delta_positive": action_mrr_delta > 0.0,
        "diffusion_action_ranking_above_random": (
            diffusion_action_mrr > random_action_mrr
        ),
        "one_step_char_accuracy_within_tolerance": one_step_char_delta >= -0.002,
        "deterministic_predicts_changed_cells": (
            deterministic["rollouts"]["next_10"]["predicted_changed_cells"] > 0
        ),
        "diffusion_predicts_changed_cells": (
            diffusion["rollouts"]["next_10"]["predicted_changed_cells"] > 0
        ),
    }
    return {
        "verdict": "supported" if all(criteria.values()) else "not_supported",
        "criteria": criteria,
        "next_10_changed_f1_delta": next_10_delta,
        "next_10_changed_f1_paired_bootstrap": interval,
        "action_ranking_mrr_delta": action_mrr_delta,
        "diffusion_action_ranking_mrr_above_random_delta": (
            diffusion_action_mrr - random_action_mrr
        ),
        "one_step_full_frame_char_accuracy_delta": one_step_char_delta,
    }


def _validate_split_arrays(arrays, split_indices) -> None:
    gameid_sets = {
        name: set(int(value) for value in arrays["gameids"][indices])
        for name, indices in split_indices.items()
    }
    if any(indices.size == 0 for indices in split_indices.values()):
        raise ValueError("train, validation, and test splits must all be non-empty")
    if gameid_sets["train"] & gameid_sets["validation"]:
        raise ValueError("train and validation game IDs overlap")
    if gameid_sets["train"] & gameid_sets["test"]:
        raise ValueError("train and test game IDs overlap")
    if gameid_sets["validation"] & gameid_sets["test"]:
        raise ValueError("validation and test game IDs overlap")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    app()
