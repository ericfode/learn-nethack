from __future__ import annotations

import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from learn_nethack.sft_train import (
    SftTrainConfig,
    apply_gradient_checkpointing_contract,
    build_assistant_mask_report,
    build_phase_rows,
    build_sft_jsonl_curriculum_plan,
    build_sft_jsonl_training_plan,
    configure_training_runtime,
    create_unsloth_sft_trainer_from_jsonl,
    format_row_for_sft,
    get_trainer_assistant_mask_report,
    require_wandb_for_training,
    resolve_jsonl_training_config,
    summarize_trainer_loss_history,
    tokenize_row_for_assistant_only_loss,
)


def _row(task: str, identifier: int) -> dict:
    return {
        "task": task,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"user {identifier}"},
            {"role": "assistant", "content": f"assistant {identifier}"},
        ],
    }


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        rendered = "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered = f"{rendered}\nassistant:"
        if tokenize:
            return list(rendered.encode("utf-8"))
        return rendered


class SftTrainTests(unittest.TestCase):
    def test_sft_train_config_defaults(self) -> None:
        config = SftTrainConfig()

        self.assertEqual(config.model_name, "google/gemma-4-E4b-it")
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.lora_r, 16)
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertFalse(config.use_gradient_checkpointing)
        self.assertEqual(config.wandb_project, "learn-nethack")
        self.assertEqual(config.torch_dynamo_recompile_limit, 128)
        self.assertTrue(config.train_on_assistant_only)
        self.assertEqual(config.dynamics_warmup_steps, 50)
        self.assertEqual(config.frame_auxiliary_ratio, 0.25)
        self.assertEqual(config.policy_calibration_steps, 20)
        self.assertEqual(config.training_objective, "policy_dynamics_phased")

    def test_configure_training_runtime_sets_wandb_project_and_dynamo_limits(
        self,
    ) -> None:
        fake_torch = types.SimpleNamespace(
            _dynamo=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    recompile_limit=8,
                    cache_size_limit=8,
                    accumulated_recompile_limit=256,
                )
            )
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.dict(os.environ, {}, clear=True),
        ):
            configure_training_runtime(SftTrainConfig(torch_dynamo_recompile_limit=64))

            self.assertEqual(os.environ["WANDB_PROJECT"], "learn-nethack")

        self.assertEqual(fake_torch._dynamo.config.recompile_limit, 64)
        self.assertEqual(fake_torch._dynamo.config.cache_size_limit, 64)
        self.assertEqual(fake_torch._dynamo.config.accumulated_recompile_limit, 256)

    def test_apply_gradient_checkpointing_contract_disables_model_state(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.disabled = False
                self.config = types.SimpleNamespace(gradient_checkpointing=True)

            def gradient_checkpointing_disable(self) -> None:
                self.disabled = True

        model = FakeModel()

        apply_gradient_checkpointing_contract(model, SftTrainConfig())

        self.assertTrue(model.disabled)
        self.assertFalse(model.config.gradient_checkpointing)

    def test_build_phase_rows_preserves_policy_calibration_and_samples_frames(
        self,
    ) -> None:
        rows = [
            _row("policy_action", 1),
            _row("policy_action", 2),
            _row("next_frame", 10),
            _row("next_frame", 11),
            _row("next_frame", 12),
            _row("next_frame", 13),
        ]
        config = SftTrainConfig(frame_auxiliary_ratio=0.5, seed=123)

        phases = build_phase_rows(rows, config)

        self.assertEqual(
            [row["task"] for row in phases["dynamics_warmup"]], ["next_frame"] * 4
        )
        self.assertEqual(
            [row["task"] for row in phases["policy_calibration"]],
            ["policy_action", "policy_action"],
        )
        self.assertEqual(
            [row["task"] for row in phases["mixed"]].count("policy_action"),
            2,
        )
        self.assertEqual(
            [row["task"] for row in phases["mixed"]].count("next_frame"),
            2,
        )
        self.assertEqual(phases, build_phase_rows(rows, config))

    def test_build_phase_rows_supports_single_objective_controls(self) -> None:
        rows = [
            _row("policy_action", 1),
            _row("policy_action", 2),
            _row("next_frame", 10),
        ]

        policy_phases = build_phase_rows(
            rows,
            SftTrainConfig(training_objective="policy_only"),
        )
        dynamics_phases = build_phase_rows(
            rows,
            SftTrainConfig(training_objective="dynamics_only"),
        )

        self.assertEqual(list(policy_phases), ["policy_only"])
        self.assertEqual(
            [row["task"] for row in policy_phases["policy_only"]],
            ["policy_action", "policy_action"],
        )
        self.assertEqual(list(dynamics_phases), ["dynamics_only"])
        self.assertEqual(
            [row["task"] for row in dynamics_phases["dynamics_only"]],
            ["next_frame"],
        )

    def test_format_row_for_sft_uses_chat_template_and_preserves_task(self) -> None:
        formatted = format_row_for_sft(_row("policy_action", 7), FakeTokenizer())

        self.assertEqual(formatted["task"], "policy_action")
        self.assertEqual(
            formatted["text"],
            "system:system\nuser:user 7\nassistant:assistant 7",
        )

    def test_assistant_only_tokenization_masks_prompt_and_keeps_response(
        self,
    ) -> None:
        tokenized = tokenize_row_for_assistant_only_loss(
            _row("policy_action", 7),
            FakeTokenizer(),
            max_seq_length=512,
        )

        prompt_count = tokenized["prompt_token_count"]
        self.assertGreater(prompt_count, 0)
        self.assertTrue(
            all(label == -100 for label in tokenized["labels"][:prompt_count])
        )
        self.assertEqual(
            tokenized["labels"][prompt_count:],
            tokenized["input_ids"][prompt_count:],
        )
        self.assertGreater(tokenized["supervised_assistant_token_count"], 0)
        self.assertEqual(tokenized["assistant_tokens_truncated"], 0)

    def test_assistant_only_tokenization_rejects_template_prefix_mismatch(
        self,
    ) -> None:
        class MismatchedTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                encoded = super().apply_chat_template(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                )
                if tokenize and not add_generation_prompt:
                    return [999, *encoded]
                return encoded

        with self.assertRaisesRegex(ValueError, "verifiable final assistant suffix"):
            tokenize_row_for_assistant_only_loss(
                _row("policy_action", 1),
                MismatchedTokenizer(),
                max_seq_length=512,
            )

    def test_assistant_only_tokenization_rejects_prompt_only_truncation(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "truncates every final assistant token"
        ):
            tokenize_row_for_assistant_only_loss(
                _row("policy_action", 1),
                FakeTokenizer(),
                max_seq_length=5,
            )

    def test_assistant_mask_report_breaks_counts_out_by_task(self) -> None:
        rows = [
            tokenize_row_for_assistant_only_loss(
                _row("policy_action", 1), FakeTokenizer(), max_seq_length=512
            ),
            tokenize_row_for_assistant_only_loss(
                _row("next_frame", 2), FakeTokenizer(), max_seq_length=512
            ),
        ]

        report = build_assistant_mask_report(rows)

        self.assertEqual(report["row_count"], 2)
        self.assertEqual(set(report["per_task"]), {"policy_action", "next_frame"})
        self.assertEqual(
            report["prompt_token_count"], report["masked_prompt_token_count"]
        )
        self.assertGreater(report["supervised_token_fraction"], 0.0)
        self.assertLess(report["supervised_token_fraction"], 1.0)

    def test_trainer_loss_history_requires_observed_decrease(self) -> None:
        report = summarize_trainer_loss_history(
            [{"loss": 3.0}, {"learning_rate": 1e-4}, {"loss": 1.25}]
        )

        self.assertEqual(report["logged_loss_count"], 2)
        self.assertEqual(report["first_loss"], 3.0)
        self.assertEqual(report["last_loss"], 1.25)
        self.assertTrue(report["loss_decreased"])

    def test_require_wandb_for_training_fails_loud_without_mode_or_key(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY is required"):
            require_wandb_for_training({})

        self.assertEqual(
            require_wandb_for_training({"WANDB_MODE": "offline"}), "offline"
        )
        self.assertEqual(
            require_wandb_for_training({"WANDB_API_KEY": "present"}), "online"
        )

    def test_jsonl_training_plan_uses_combined_rows_and_adapter_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "sft-data"
            dataset_dir.mkdir()
            (dataset_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
            output_dir = root / "adapters"

            plan = build_sft_jsonl_training_plan(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                config=SftTrainConfig(model_name="google/gemma-4-E4b-it"),
            )

        self.assertEqual(
            plan["schema_version"], "learn-nethack.sft-jsonl-train-plan.v1"
        )
        self.assertEqual(plan["train_files"], [str(dataset_dir / "train.jsonl")])
        self.assertEqual(plan["output_dir"], str(output_dir))
        self.assertEqual(plan["model_name"], "google/gemma-4-E4b-it")
        self.assertEqual(plan["tasks"], ["policy_action", "next_frame"])

    def test_jsonl_training_plan_reads_manifest_tasks_when_available(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "sft-data"
            dataset_dir.mkdir()
            (dataset_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
            (dataset_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.sft-manifest.v1",
                        "tasks": ["policy_action"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            plan = build_sft_jsonl_training_plan(
                dataset_dir=dataset_dir,
                output_dir=root / "adapters",
                config=SftTrainConfig(),
            )

        self.assertEqual(plan["tasks"], ["policy_action"])

    def test_jsonl_curriculum_plan_uses_task_files_and_sampled_frames(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "sft-data"
            scratch_dir = root / "scratch"
            dataset_dir.mkdir()
            policy_lines = [json.dumps(_row("policy_action", i)) for i in range(8)]
            frame_lines = [json.dumps(_row("next_frame", i)) for i in range(8)]
            (dataset_dir / "train.policy_action.jsonl").write_text(
                "\n".join(policy_lines) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "train.next_frame.jsonl").write_text(
                "\n".join(frame_lines) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(policy_lines + frame_lines) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "learn-nethack.sft-manifest.v1",
                        "tasks": ["policy_action", "next_frame"],
                    }
                ),
                encoding="utf-8",
            )

            plan = build_sft_jsonl_curriculum_plan(
                dataset_dir=dataset_dir,
                output_dir=root / "adapters",
                scratch_dir=scratch_dir,
                config=SftTrainConfig(
                    max_steps=250,
                    frame_auxiliary_ratio=0.5,
                    seed=7,
                ),
                requested_max_steps=250,
            )
            mixed_path = Path(plan["phases"][1]["train_files"][0])
            mixed_file_exists = mixed_path.exists()
            mixed_rows = [
                json.loads(line)
                for line in mixed_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(
            plan["schema_version"], "learn-nethack.sft-jsonl-curriculum-plan.v1"
        )
        self.assertEqual(plan["curriculum"], "policy_dynamics_phased")
        self.assertEqual(
            [phase["name"] for phase in plan["phases"]],
            ["dynamics_warmup", "mixed", "policy_calibration"],
        )
        self.assertEqual(
            [phase["max_steps"] for phase in plan["phases"]],
            [50, 180, 20],
        )
        self.assertEqual(plan["policy_row_count"], 8)
        self.assertEqual(plan["next_frame_row_count"], 8)
        self.assertGreater(plan["sampled_next_frame_rows"], 0)
        self.assertLess(plan["sampled_next_frame_rows"], 8)
        self.assertTrue(mixed_file_exists)
        self.assertTrue(all(set(row) == {"messages", "task"} for row in mixed_rows))
        self.assertEqual(
            len(mixed_rows),
            plan["policy_row_count"] + plan["sampled_next_frame_rows"],
        )

    def test_jsonl_curriculum_plan_can_train_policy_only_control(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "sft-data"
            dataset_dir.mkdir()
            policy_lines = [json.dumps(_row("policy_action", i)) for i in range(6)]
            frame_lines = [json.dumps(_row("next_frame", i)) for i in range(4)]
            (dataset_dir / "train.policy_action.jsonl").write_text(
                "\n".join(policy_lines) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "train.next_frame.jsonl").write_text(
                "\n".join(frame_lines) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(policy_lines + frame_lines) + "\n",
                encoding="utf-8",
            )

            plan = build_sft_jsonl_curriculum_plan(
                dataset_dir=dataset_dir,
                output_dir=root / "adapters",
                scratch_dir=root / "scratch",
                config=SftTrainConfig(
                    max_steps=30,
                    training_objective="policy_only",
                ),
                requested_max_steps=30,
            )

        self.assertEqual(plan["curriculum"], "policy_only")
        self.assertEqual(plan["training_objective"], "policy_only")
        self.assertEqual(plan["policy_row_count"], 6)
        self.assertEqual(plan["next_frame_row_count"], 0)
        self.assertEqual(plan["phases"][0]["tasks"], ["policy_action"])
        self.assertEqual(plan["phases"][0]["max_steps"], 30)

    def test_jsonl_curriculum_plan_can_train_dynamics_only_control(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "sft-data"
            dataset_dir.mkdir()
            policy_lines = [json.dumps(_row("policy_action", i)) for i in range(6)]
            frame_lines = [json.dumps(_row("next_frame", i)) for i in range(7)]
            (dataset_dir / "train.policy_action.jsonl").write_text(
                "\n".join(policy_lines) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "train.next_frame.jsonl").write_text(
                "\n".join(frame_lines) + "\n",
                encoding="utf-8",
            )
            (dataset_dir / "train.jsonl").write_text(
                "\n".join(policy_lines + frame_lines) + "\n",
                encoding="utf-8",
            )

            plan = build_sft_jsonl_curriculum_plan(
                dataset_dir=dataset_dir,
                output_dir=root / "adapters",
                scratch_dir=root / "scratch",
                config=SftTrainConfig(
                    max_steps=0,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    training_objective="dynamics_only",
                ),
                requested_max_steps=0,
            )

        self.assertEqual(plan["curriculum"], "dynamics_only")
        self.assertEqual(plan["training_objective"], "dynamics_only")
        self.assertEqual(plan["policy_row_count"], 0)
        self.assertEqual(plan["next_frame_row_count"], 7)
        self.assertEqual(plan["phases"][0]["tasks"], ["next_frame"])
        self.assertEqual(plan["phases"][0]["max_steps"], 2)

    def test_jsonl_curriculum_plan_auto_steps_cover_balanced_mixed_rows(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "sft-data"
            dataset_dir.mkdir()
            policy_lines = [json.dumps(_row("policy_action", i)) for i in range(9)]
            frame_lines = [json.dumps(_row("next_frame", i)) for i in range(9)]
            for name, lines in (
                ("train.policy_action.jsonl", policy_lines),
                ("train.next_frame.jsonl", frame_lines),
                ("train.jsonl", policy_lines + frame_lines),
            ):
                (dataset_dir / name).write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )

            plan = build_sft_jsonl_curriculum_plan(
                dataset_dir=dataset_dir,
                output_dir=root / "adapters",
                scratch_dir=root / "scratch",
                config=SftTrainConfig(
                    max_steps=5,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    frame_auxiliary_ratio=1.0,
                ),
                requested_max_steps=0,
            )

        self.assertEqual(plan["max_steps_mode"], "auto_curriculum_one_mixed_pass")
        self.assertEqual(plan["effective_batch_size"], 4)
        self.assertEqual(plan["sampled_next_frame_rows"], 9)
        self.assertEqual(
            [phase["max_steps"] for phase in plan["phases"]],
            [3, 5, 3],
        )

    def test_jsonl_training_plan_requires_train_jsonl(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "train.jsonl"):
                build_sft_jsonl_training_plan(
                    dataset_dir=Path(tmp) / "missing",
                    output_dir=Path(tmp) / "adapters",
                    config=SftTrainConfig(),
                )

    def test_resolve_jsonl_training_config_keeps_explicit_steps(self) -> None:
        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "train.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            config = SftTrainConfig(max_steps=7)

            resolved = resolve_jsonl_training_config(
                dataset_dir=dataset_dir,
                config=config,
            )

        self.assertIs(resolved.config, config)
        self.assertEqual(resolved.max_steps_mode, "explicit")
        self.assertEqual(resolved.train_row_count, 2)
        self.assertEqual(resolved.effective_batch_size, 4)

    def test_resolve_jsonl_training_config_auto_steps_cover_full_train_file(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "train.jsonl").write_text("{}\n" * 9, encoding="utf-8")
            config = SftTrainConfig(
                max_steps=0,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
            )

            resolved = resolve_jsonl_training_config(
                dataset_dir=dataset_dir,
                config=config,
            )

        self.assertEqual(resolved.config.max_steps, 3)
        self.assertEqual(resolved.max_steps_mode, "auto_one_full_pass")
        self.assertEqual(resolved.train_row_count, 9)
        self.assertEqual(resolved.effective_batch_size, 4)

    def test_resolve_jsonl_training_config_auto_steps_use_objective_file(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "train.jsonl").write_text("{}\n" * 12, encoding="utf-8")
            (dataset_dir / "train.policy_action.jsonl").write_text(
                "{}\n" * 5,
                encoding="utf-8",
            )
            config = SftTrainConfig(
                max_steps=0,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                training_objective="policy_only",
            )

            resolved = resolve_jsonl_training_config(
                dataset_dir=dataset_dir,
                config=config,
            )

        self.assertEqual(resolved.config.max_steps, 2)
        self.assertEqual(resolved.max_steps_mode, "auto_one_full_pass")
        self.assertEqual(resolved.train_row_count, 5)
        self.assertEqual(resolved.effective_batch_size, 4)

    def test_jsonl_trainer_gates_wandb_before_heavy_imports(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY is required"):
            create_unsloth_sft_trainer_from_jsonl(
                jsonl_paths=[Path("missing-train.jsonl")],
                output_dir=Path("missing-adapters"),
                config=SftTrainConfig(),
                env={},
            )

    def test_jsonl_trainer_uses_explicit_assistant_only_labels(
        self,
    ) -> None:
        captured_args: dict = {}
        captured_peft_args: dict = {}

        class FakeFastLanguageModel:
            @staticmethod
            def from_pretrained(**_kwargs):
                return object(), FakeTokenizer()

            @staticmethod
            def get_peft_model(model, **kwargs):
                captured_peft_args.update(kwargs)
                return model

        class FakeDataset:
            column_names = ["messages", "task"]

            def map(
                self,
                formatter,
                *,
                batched,
                remove_columns,
                load_from_cache_file,
            ):
                self.formatted = formatter(
                    {
                        "messages": [_row("policy_action", 1)["messages"]],
                        "task": ["policy_action"],
                    }
                )
                self.batched = batched
                self.remove_columns = remove_columns
                self.load_from_cache_file = load_from_cache_file
                return self

        def fake_load_dataset(*_args, **_kwargs):
            return FakeDataset()

        class FakeSFTConfig:
            def __init__(self, **kwargs):
                captured_args.update(kwargs)

        class FakeSFTTrainer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        modules = {
            "unsloth": types.SimpleNamespace(FastLanguageModel=FakeFastLanguageModel),
            "datasets": types.SimpleNamespace(load_dataset=fake_load_dataset),
            "trl": types.SimpleNamespace(
                SFTConfig=FakeSFTConfig,
                SFTTrainer=FakeSFTTrainer,
            ),
        }

        with TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            train_path = Path(tmp) / "train.jsonl"
            train_path.write_text(
                json.dumps(_row("policy_action", 1)), encoding="utf-8"
            )

            trainer = create_unsloth_sft_trainer_from_jsonl(
                jsonl_paths=[train_path],
                output_dir=Path(tmp) / "adapter",
                config=SftTrainConfig(),
                env={"WANDB_MODE": "offline"},
            )

        self.assertFalse(captured_args["assistant_only_loss"])
        self.assertFalse(captured_args["completion_only_loss"])
        self.assertFalse(captured_args["gradient_checkpointing"])
        self.assertIsNone(captured_args["dataset_num_proc"])
        self.assertFalse(captured_peft_args["use_gradient_checkpointing"])
        dataset = trainer.kwargs["train_dataset"]
        prompt_count = dataset.formatted["prompt_token_count"][0]
        labels = dataset.formatted["labels"][0]
        self.assertTrue(all(label == -100 for label in labels[:prompt_count]))
        self.assertTrue(all(label != -100 for label in labels[prompt_count:]))
        self.assertFalse(dataset.load_from_cache_file)
        mask_report = get_trainer_assistant_mask_report(trainer)
        self.assertEqual(mask_report["row_count"], 1)
        self.assertGreater(mask_report["supervised_assistant_token_count"], 0)


if __name__ == "__main__":
    unittest.main()
