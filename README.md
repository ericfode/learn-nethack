# learn-nethack

Reproducible Gemma/NetHack learning pipeline.

The current checked-in lanes are:

- Local NLD SFT data loop for multi-task `policy_action` and `next_frame` rows.
- Modal readiness for later GPU training and W&B-backed run artifacts.

Raw NetHack data lives outside git. Generated artifacts belong under
`artifacts/`.

## Local NLD SFT Data Loop

Inspect the local NLD-AA taster metadata:

```bash
uv run nethack-gemma data inspect \
  --db /Users/ericfode/data/nld/nld-aa-taster/ttyrecs.db
```

Write a build-contract smoke without decoding ttyrecs:

```bash
uv run nethack-gemma data write-build-contract \
  --db /Users/ericfode/data/nld/nld-aa-taster/ttyrecs.db \
  --mode single_frame \
  --tasks policy_action,next_frame \
  --max-rows 64 \
  --out artifacts/sft/nld-aa-taster-contract-smoke
```

When `nle.dataset` is installed and an action manifest exists, build SFT rows:

```bash
uv run --extra local-nle nethack-gemma data write-action-manifest \
  --out artifacts/action_manifest.json

WANDB_MODE=offline uv run --extra local-nle nethack-gemma data build-sft \
  --db /Users/ericfode/data/nld/nld-aa-taster/ttyrecs.db \
  --action-manifest artifacts/action_manifest.json \
  --mode single_frame \
  --tasks policy_action,next_frame \
  --seq-length 128 \
  --batch-size 4 \
  --max-rows 1000 \
  --out artifacts/sft/nld-aa-taster-single-frame-smoke
```

Unset `WANDB_MODE=offline` only when `WANDB_API_KEY` is available and an online
W&B run is intended. The build writes local reports first, then mirrors the
metrics and manifest artifacts to W&B.

Build every accepted row from the currently indexed local taster DB:

```bash
WANDB_MODE=offline uv run --extra local-nle nethack-gemma data build-sft \
  --db /Users/ericfode/data/nld/nld-aa-taster/ttyrecs.db \
  --action-manifest artifacts/action_manifest.json \
  --mode single_frame \
  --tasks policy_action,next_frame \
  --seq-length 128 \
  --batch-size 4 \
  --full-dataset \
  --out artifacts/sft/nld-aa-taster-single-frame-full
```

The larger local NAO archive tree under
`/Users/ericfode/data/nld/history/nld-nao` is not indexed yet. Build an NLE DB
artifact for it without modifying the source corpus:

```bash
uv run --extra local-nle nethack-gemma data index-altorg \
  --metadata-root /Users/ericfode/data/nld/history/nld-nao/unpacked \
  --ttyrec-root /Users/ericfode/data/nld/history/nld-nao/unpacked/nld-nao-unzipped \
  --staging-root artifacts/nld-index/nld-nao-staging \
  --db artifacts/nld-index/nld-nao-ttyrecs.db \
  --dataset-name nld-nao
```

The builder writes combined and task-specific splits:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`
- `train.policy_action.jsonl`
- `train.next_frame.jsonl`
- `manifest.json`
- `rejection_report.json`

## Learned Dynamics Play

Use the `play dynamics` command to interact with a Gemma adapter trained on the
`next_frame` task. This is not a live NLE game. It is a learned dynamics loop:
you provide an `action_id`, the model predicts `{"next_frame": "..."}`, and that
predicted frame becomes the next prompt.

Write the command contract without loading model dependencies:

```bash
uv run nethack-gemma play dynamics \
  --action-manifest artifacts/action_manifest.json \
  --adapter-checkpoint artifacts/runs/dynamics-adapter \
  --initial-row artifacts/sft/nld-aa-taster-single-frame-smoke/validation.next_frame.jsonl \
  --out artifacts/play/dynamics-contract-smoke \
  --dry-run-contract
```

Run an interactive terminal session when the adapter checkpoint is available:

```bash
uv run --extra modal-train nethack-gemma play dynamics \
  --action-manifest artifacts/action_manifest.json \
  --adapter-checkpoint /path/to/next-frame-adapter \
  --initial-row artifacts/sft/nld-aa-taster-single-frame-smoke/validation.next_frame.jsonl \
  --out artifacts/play/dynamics-session
```

Run a deterministic scripted session:

```bash
uv run --extra modal-train nethack-gemma play dynamics \
  --action-manifest artifacts/action_manifest.json \
  --adapter-checkpoint /path/to/next-frame-adapter \
  --initial-row artifacts/sft/nld-aa-taster-single-frame-smoke/validation.next_frame.jsonl \
  --actions 1,0,1 \
  --out artifacts/play/dynamics-scripted
```

Every session writes `events.jsonl`, `latest.json`, `report.json`, and
`index.html` under the selected output directory. Invalid model output stops the
session with a `parse_failed` event instead of silently advancing state.

## Modal Readiness Smoke

Run an explicitly offline smoke when testing cloud plumbing without W&B
credentials or the training secret:

```bash
WANDB_MODE=offline modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-smoke
```

The report must show `"execution": {"backend": "modal_cloud", ...}`. A local
static import is not sufficient evidence of Modal readiness.

This workspace already expects two Modal secrets:

```bash
modal secret list
modal secret create hf-token HF_TOKEN="$HF_TOKEN"
modal secret create wandb-secret WANDB_API_KEY="$WANDB_API_KEY"
```

Run a credentialed readiness smoke:

```bash
modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-smoke
```

## Modal SFT Proof Loop

Upload the local taster corpus and action manifest into the dataset volume:

```bash
modal volume put learn-nethack-datasets /Users/ericfode/data/nld/nld-aa-taster /nld/
modal volume put learn-nethack-datasets artifacts/action_manifest.json /action_manifest.json
```

Record a baseline validation score:

```bash
modal run src/learn_nethack/modal_train.py::sft_eval \
  --run-id full-sft-baseline \
  --db /datasets/nld/nld-aa-taster/ttyrecs.db \
  --action-manifest /datasets/action_manifest.json \
  --nle-root /datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data \
  --split validation \
  --model-role baseline
```

Fine-tune on every accepted row from the uploaded dataset:

```bash
modal run src/learn_nethack/modal_train.py::sft_train \
  --run-id full-sft \
  --db /datasets/nld/nld-aa-taster/ttyrecs.db \
  --action-manifest /datasets/action_manifest.json \
  --nle-root /datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data \
  --mode single_frame \
  --full-dataset
```

Evaluate the adapter and compare it to the baseline:

```bash
modal run src/learn_nethack/modal_train.py::sft_eval \
  --run-id full-sft-trained \
  --db /datasets/nld/nld-aa-taster/ttyrecs.db \
  --action-manifest /datasets/action_manifest.json \
  --nle-root /datasets/nld/nld-aa-taster/unpacked/nld-aa-taster/nle_data \
  --split validation \
  --adapter /runs/full-sft/adapters \
  --model-role trained

modal run src/learn_nethack/modal_train.py::sft_compare \
  --baseline /runs/full-sft-baseline/reports/sft_eval_metrics.json \
  --trained /runs/full-sft-trained/reports/sft_eval_metrics.json \
  --out /runs/full-sft/reports/score_to_beat.json \
  --trained-run-id full-sft-trained \
  --baseline-run-id full-sft-baseline
```

Generated reports and W&B files belong under Modal volumes and local
`artifacts/`; raw NetHack data, secrets, checkpoints, ttyrecs, and media stay
out of git.

## Side-By-Side Watch Harness

Compare a current LoRA checkpoint against baseline Gemma in matching seeded NLE
environments:

```bash
uv run --extra watch nethack-gemma watch compare \
  --action-manifest artifacts/action_manifest.json \
  --out artifacts/watch/compare-smoke \
  --run-id compare-smoke \
  --max-steps 80
```

Add `--current-checkpoint /path/to/current/adapter` when a local LoRA adapter is
available. Without it, the harness compares base Gemma against base Gemma; this
is useful for dependency and watch-surface smokes.

The harness writes `events.jsonl`, `latest.json`, `report.json`, and
`index.html` under the output directory. It scores exact `{"action_id": N}`
candidates from the active action manifest; it does not use free-text action
generation.
