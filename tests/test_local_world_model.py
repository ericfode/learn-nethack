from __future__ import annotations

import json

import numpy as np
import pytest

from learn_nethack.action_manifest import ActionEntry, ActionManifest
from learn_nethack.local_world_model import _build_comparison
from learn_nethack.world_model_aggregate import aggregate_world_model_reports
from learn_nethack.world_model_data import collect_terminal_transition_arrays
from learn_nethack.world_model_metrics import (
    apply_terminal_delta,
    contiguous_rollout_starts,
    frame_metric_sums,
    paired_bootstrap_interval,
    terminal_delta,
)
from learn_nethack.world_model_watch import write_world_model_watch


def test_terminal_delta_round_trips_byte_planes() -> None:
    current = np.asarray([[32, 64], [10, 255]], dtype=np.uint8)
    following = np.asarray([[32, 65], [0, 255]], dtype=np.uint8)

    delta = terminal_delta(current, following)

    assert delta.tolist() == [[0, 66], [1, 0]]
    assert np.array_equal(apply_terminal_delta(current, delta), following)


def test_frame_metrics_do_not_reward_copying_changed_cells() -> None:
    current_chars = np.full((1, 24, 80), 32, dtype=np.uint8)
    current_colors = np.zeros((1, 24, 80), dtype=np.uint8)
    truth_chars = current_chars.copy()
    truth_chars[0, 10, 10] = ord("@")
    copied_chars = current_chars.copy()

    metrics = frame_metric_sums(
        (current_chars, current_colors),
        (truth_chars, current_colors),
        (copied_chars, current_colors),
    ).as_metrics()

    assert metrics["full_frame_char_accuracy"] > 0.99
    assert metrics["changed_cell_f1"] == 0.0
    assert metrics["changed_cell_char_accuracy"] == 0.0


def test_contiguous_rollout_starts_reject_boundaries_and_gaps() -> None:
    sequence_ids = np.asarray([1, 1, 1, 2, 2, 2, 2])
    sequence_steps = np.asarray([0, 1, 3, 0, 1, 2, 3])
    candidates = np.arange(sequence_ids.size)

    starts = contiguous_rollout_starts(
        sequence_ids,
        sequence_steps,
        candidates,
        horizon=3,
    )

    assert starts.tolist() == [3, 4]


def test_paired_bootstrap_detects_uniform_positive_difference() -> None:
    interval = paired_bootstrap_interval(
        np.zeros(32),
        np.full(32, 0.2),
        seed=7,
        samples=200,
    )

    assert interval["mean_difference"] == pytest.approx(0.2)
    assert interval["lower"] > 0.0


def test_collect_terminal_arrays_preserves_disjoint_split_and_sequence() -> None:
    batch = _fake_sequence_batch()
    manifest = ActionManifest(
        env_id="fixture",
        entries=(
            ActionEntry(
                action_id=3,
                nle_action_name="fixture.west",
                raw_key_code=104,
                key_label="h",
            ),
        ),
    )

    arrays, counts, rejections = collect_terminal_transition_arrays(
        batches=[batch],
        action_manifest=manifest,
        split_gameids_by_name={
            "train": {11},
            "validation": {22},
            "test": {33},
        },
        transition_limits={"train": 2, "validation": 2, "test": 2},
    )

    assert counts == {"train": 2, "validation": 2, "test": 2}
    assert not rejections
    assert arrays["current_chars"].shape == (6, 24, 80)
    assert arrays["action_ids"].tolist() == [3] * 6
    assert arrays["split_codes"].tolist() == [0, 0, 1, 1, 2, 2]
    assert arrays["sequence_steps"].tolist() == [0, 1, 0, 1, 0, 1]
    assert len(set(arrays["sequence_ids"].tolist())) == 3


def test_matched_models_have_equal_parameters_and_finite_losses() -> None:
    torch = pytest.importorskip("torch")
    from learn_nethack._world_model_torch import (
        TerminalWorldModelConfig,
        build_model,
        parameter_count,
        training_loss,
    )

    config = TerminalWorldModelConfig(
        action_vocab_size=8,
        hidden_channels=16,
        residual_blocks=1,
        diffusion_steps=2,
    )
    deterministic = build_model(config, seed=4)
    diffusion = build_model(config, seed=4)
    batch = {
        "current_chars": torch.full((2, 24, 80), 32, dtype=torch.long),
        "current_colors": torch.zeros((2, 24, 80), dtype=torch.long),
        "next_chars": torch.full((2, 24, 80), 32, dtype=torch.long),
        "next_colors": torch.zeros((2, 24, 80), dtype=torch.long),
        "action_ids": torch.tensor([1, 2], dtype=torch.long),
    }
    batch["next_chars"][0, 4, 5] = ord("@")

    deterministic_loss, _ = training_loss(
        deterministic,
        batch,
        variant="deterministic",
        config=config,
    )
    diffusion_loss, _ = training_loss(
        diffusion,
        batch,
        variant="diffusion",
        config=config,
    )

    assert parameter_count(deterministic) == parameter_count(diffusion)
    assert torch.isfinite(deterministic_loss)
    assert torch.isfinite(diffusion_loss)


def test_pre_registered_comparison_requires_every_gate() -> None:
    deterministic = _evaluation_fixture(
        next_10_f1=0.20,
        action_mrr=0.30,
        char_accuracy=0.99,
    )
    diffusion = _evaluation_fixture(
        next_10_f1=0.25,
        action_mrr=0.35,
        char_accuracy=0.989,
    )

    comparison = _build_comparison(
        deterministic=deterministic,
        diffusion=diffusion,
        deterministic_next_10=np.full(64, 0.20),
        diffusion_next_10=np.full(64, 0.25),
        seed=9,
    )

    assert comparison["verdict"] == "supported"
    diffusion["action_ranking"]["mean_reciprocal_rank"] = 0.29
    failed = _build_comparison(
        deterministic=deterministic,
        diffusion=diffusion,
        deterministic_next_10=np.full(64, 0.20),
        diffusion_next_10=np.full(64, 0.25),
        seed=9,
    )
    assert failed["verdict"] == "not_supported"


def test_world_model_watch_emits_valid_jsonl_split_regex(tmp_path) -> None:
    write_world_model_watch(
        out_dir=tmp_path,
        run_id="fixture",
        events=[{"event_index": 0}],
    )

    html = (tmp_path / "index.html").read_text(encoding="utf-8")

    assert "split(/\\n+/)" in html
    assert "split(/\n" not in html


def test_aggregate_world_model_reports_preserves_seed_deltas(tmp_path) -> None:
    report_paths = []
    for offset, (deterministic_value, diffusion_value) in enumerate(
        ((0.7, 0.6), (0.8, 0.9))
    ):
        contract_path = tmp_path / f"contract-{offset}.json"
        report_path = tmp_path / f"report-{offset}.json"
        contract_path.write_text(
            json.dumps(
                {
                    "dataset_sha256": "same-data",
                    "model_config": {"hidden_channels": 8},
                    "matched_parameter_count": 42,
                    "train_config": {"seed": offset + 1},
                    "eval_config": {"seed": 99},
                }
            ),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": "learn-nethack.local-world-model-proof.v1",
                    "run_id": f"run-{offset}",
                    "contract_path": str(contract_path),
                    "verdict": "not_supported",
                    "wandb": {"run_id": f"wandb-{offset}"},
                    "evaluation": {
                        "deterministic": _aggregate_variant_fixture(
                            deterministic_value
                        ),
                        "diffusion": _aggregate_variant_fixture(diffusion_value),
                    },
                    "comparison": {
                        "next_10_changed_f1_delta": (
                            diffusion_value - deterministic_value
                        ),
                        "next_10_changed_f1_paired_bootstrap": {
                            "mean_difference": diffusion_value - deterministic_value
                        },
                        "action_ranking_mrr_delta": (
                            diffusion_value - deterministic_value
                        ),
                        "one_step_full_frame_char_accuracy_delta": (
                            diffusion_value - deterministic_value
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        report_paths.append(report_path)

    aggregate = aggregate_world_model_reports(
        report_paths=report_paths,
        out_path=tmp_path / "aggregate.json",
    )

    assert aggregate["run_count"] == 2
    assert aggregate["verdict"] == "not_supported"
    assert aggregate["diffusion_minus_deterministic"]["action_mrr"][
        "mean"
    ] == pytest.approx(0.0)


def _fake_sequence_batch() -> dict[str, np.ndarray]:
    chars = np.full((3, 3, 24, 80), 32, dtype=np.uint8)
    colors = np.zeros_like(chars)
    for batch_index in range(3):
        chars[batch_index, 1, 10, 10] = ord("@")
        chars[batch_index, 2, 10, 11] = ord("@")
    return {
        "tty_chars": chars,
        "tty_colors": colors,
        "tty_cursor": np.zeros((3, 3, 2), dtype=np.int16),
        "timestamps": np.asarray(
            [[100, 101, 102], [200, 201, 202], [300, 301, 302]],
            dtype=np.int64,
        ),
        "done": np.zeros((3, 3), dtype=bool),
        "gameids": np.asarray([[11] * 3, [22] * 3, [33] * 3]),
        "keypresses": np.full((3, 3), 104, dtype=np.int16),
    }


def _evaluation_fixture(
    *,
    next_10_f1: float,
    action_mrr: float,
    char_accuracy: float,
) -> dict:
    return {
        "one_step": {"full_frame_char_accuracy": char_accuracy},
        "rollouts": {
            "next_10": {
                "changed_cell_f1": next_10_f1,
                "predicted_changed_cells": 10,
            }
        },
        "action_ranking": {
            "mean_reciprocal_rank": action_mrr,
            "random_mean_reciprocal_rank": 0.3397321428571428,
        },
    }


def _aggregate_variant_fixture(value: float) -> dict:
    return {
        "one_step": {
            "changed_cell_f1": value,
            "full_frame_char_accuracy": value,
        },
        "rollouts": {
            horizon: {
                "changed_cell_f1": value,
                "full_frame_char_accuracy": value,
            }
            for horizon in ("next_1", "next_5", "next_10")
        },
        "action_ranking": {
            "mean_reciprocal_rank": value,
            "top1_accuracy": value,
        },
    }
