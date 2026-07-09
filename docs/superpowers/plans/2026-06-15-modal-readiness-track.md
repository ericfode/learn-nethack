# Modal Readiness Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Modal ready for Gemma/NetHack SFT, eval, and RL training without touching the baseline-evaluation lane.

**Architecture:** Keep Modal readiness as a narrow contract layer: static resource names and gates live in `src/learn_nethack/modal_config.py`, while `src/learn_nethack/modal_train.py` owns the Modal app and an initial readiness entrypoint. Training code can later attach SFT/eval/RL functions to the same app, volumes, secrets, artifact layout, and W&B contract without importing BALROG into the trainer path.

**Tech Stack:** Python 3.11, Modal, W&B, PyTorch, Transformers, TRL, Unsloth, NLE, FastAPI/WebSocket for later watch serving, optional separate vLLM image support. BALROG remains an external evaluation/reference harness, not a Modal training dependency.

---

## Current State

This track starts from an unborn `main` branch containing only repository guidance and plans. The initial Modal readiness patch creates:

- `pyproject.toml`: package metadata, core `wandb` dependency, and optional `modal-train` / `modal-vllm` dependency groups.
- `.gitignore`: generated artifacts, W&B state, Modal state, checkpoints, ttyrecs, media, and env files excluded from git.
- `README.md`: Modal readiness commands and data/secrets boundary.
- `src/learn_nethack/modal_config.py`: static Modal resource contract, W&B mode resolver, artifact layout, and smoke commands.
- `src/learn_nethack/modal_train.py`: Modal app factory and `readiness` entrypoint.
- `tests/test_modal_readiness.py`: local static gate requiring no Modal network access.

The baseline-evaluation thread remains separate. This lane does not add BALROG to the trainer dependency path and does not define baseline scoring behavior.

## File Structure

- Modify `src/learn_nethack/modal_config.py`: single source of truth for Modal app name, volumes, secret env vars, image packages, artifact paths, smoke commands, and W&B mode validation.
- Modify `src/learn_nethack/modal_train.py`: Modal app, image, mounted volumes, secret binding, readiness entrypoint, and future SFT/eval/RL function attachment.
- Modify `tests/test_modal_readiness.py`: local assertions for dependency boundaries, resource names, W&B gates, artifact layout, and exact smoke commands.
- Modify `README.md`: operator-facing Modal setup commands.
- Create later `docs/superpowers/reports/YYYY-MM-DD-modal-readiness-smoke.md`: actual Modal smoke result with report paths and W&B run URL or offline run directory.

## Resource Contract

Modal app:

```text
learn-nethack-gemma
```

Modal secrets:

```text
hf-token       -> HF_TOKEN
wandb-secret  -> WANDB_API_KEY
```

Required secret environment variables:

```text
HF_TOKEN
WANDB_API_KEY
```

Modal volumes:

```text
learn-nethack-datasets   -> /datasets
learn-nethack-runs       -> /runs
learn-nethack-hf-cache   -> /cache/huggingface
learn-nethack-watch      -> /watch
```

Canonical run artifact layout for `run_id=modal-readiness-smoke`:

```text
/runs/modal-readiness-smoke/reports/modal_readiness_report.json
/runs/modal-readiness-smoke/wandb
/runs/modal-readiness-smoke/adapters
/runs/modal-readiness-smoke/ttyrec
/runs/modal-readiness-smoke/replay
/watch/modal-readiness-smoke
```

W&B gate:

- Online Modal SFT/eval/RL runs require `WANDB_API_KEY`.
- Credentialless runs must set `WANDB_MODE=offline` explicitly.
- Every Modal run writes the local JSON report before or alongside W&B logging.

## Tasks

### Task 1: Verify Local Modal Contract

**Files:**
- Read: `pyproject.toml`
- Read: `.gitignore`
- Read: `src/learn_nethack/modal_config.py`
- Read: `src/learn_nethack/modal_train.py`
- Test: `tests/test_modal_readiness.py`

- [ ] **Step 1: Run the static unittest gate**

```bash
PYTHONPATH=src python3 -m unittest tests/test_modal_readiness.py -q
```

Expected:

```text
Ran 6 tests
OK
```

- [ ] **Step 2: Run the pytest gate after dependencies are synced**

```bash
uv run pytest tests/test_modal_readiness.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 3: Verify no BALROG dependency entered the trainer path**

```bash
rg -n "balrog|BALROG" pyproject.toml src tests
```

Expected: no match in `pyproject.toml` or `src/learn_nethack/modal_config.py`. Matches in documentation or comments are allowed only when they preserve the boundary.

### Task 2: Create Modal External Resources

**Files:**
- No repository file changes.

- [ ] **Step 1: Authenticate Modal**

```bash
modal token set
```

Expected: Modal stores credentials outside the repository.

- [ ] **Step 2: Verify or create the training secrets**

```bash
modal secret list
modal secret create hf-token HF_TOKEN="$HF_TOKEN"
modal secret create wandb-secret WANDB_API_KEY="$WANDB_API_KEY"
```

Expected: Modal shows or creates the two secrets. Literal token values must not be written to any repository file.

- [ ] **Step 3: Create the datasets volume**

```bash
modal volume create learn-nethack-datasets
```

Expected: volume exists or Modal reports that it already exists.

- [ ] **Step 4: Create the runs volume**

```bash
modal volume create learn-nethack-runs
```

Expected: volume exists or Modal reports that it already exists.

- [ ] **Step 5: Create the Hugging Face cache volume**

```bash
modal volume create learn-nethack-hf-cache
```

Expected: volume exists or Modal reports that it already exists.

- [ ] **Step 6: Create the watcher volume**

```bash
modal volume create learn-nethack-watch
```

Expected: volume exists or Modal reports that it already exists.

### Task 3: Run Offline Modal Readiness Smoke

**Files:**
- Read: `src/learn_nethack/modal_train.py`
- Produced outside git: `/runs/modal-readiness-offline-20260615/reports/modal_readiness_report.json`
- Produced outside git: `/runs/modal-readiness-offline-20260615/wandb`

- [ ] **Step 1: Run the explicit offline smoke**

```bash
WANDB_MODE=offline modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-offline-20260615
```

Expected:

- Modal builds the GPU image.
- The function writes `/runs/modal-readiness-offline-20260615/reports/modal_readiness_report.json`.
- The report contains `"mode": "offline"` under `"wandb"`.
- W&B writes an offline run directory under `/runs/modal-readiness-offline-20260615/wandb`.

- [ ] **Step 2: Record the smoke result**

Create `docs/superpowers/reports/2026-06-15-modal-readiness-smoke.md` with:

```markdown
# Modal Readiness Smoke - 2026-06-15

## Command

`WANDB_MODE=offline modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-offline-20260615`

## Result

- Status: pass
- Report: `/runs/modal-readiness-offline-20260615/reports/modal_readiness_report.json`
- W&B: `/runs/modal-readiness-offline-20260615/wandb`
- Notes: offline W&B was explicit; no secrets were written to git.
```

### Task 4: Run Online Modal Readiness Smoke

**Files:**
- Read: `src/learn_nethack/modal_train.py`
- Produced outside git: `/runs/modal-readiness-online-20260615/reports/modal_readiness_report.json`
- Produced outside git: W&B online run under project `learn-nethack`

- [ ] **Step 1: Run the credentialed smoke**

```bash
modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-online-20260615
```

Expected:

- Modal injects `hf-token` and `wandb-secret`.
- `resolve_wandb_mode` selects online mode.
- The local JSON report is written before or alongside W&B logging.
- W&B creates an online run named `modal-readiness-online-20260615`.

- [ ] **Step 2: Record the online W&B evidence**

Append this section to `docs/superpowers/reports/2026-06-15-modal-readiness-smoke.md` only after the W&B URL is known:

```markdown
## Online W&B Smoke

- Command: `modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-online-20260615`
- Status: pass
- Report: `/runs/modal-readiness-online-20260615/reports/modal_readiness_report.json`
```

Then add one more line beginning `- W&B URL: ` followed by the full `https://` URL from Modal output. Do not write a synthetic URL and do not paste API keys.

### Task 5: Attach Future Trainer Entrypoints To This App

**Files:**
- Modify: `src/learn_nethack/modal_train.py`
- Modify: `tests/test_modal_readiness.py` or create focused trainer config tests.

- [ ] **Step 1: Add tests for SFT/eval/RL function names before implementation**

Add assertions that `modal_train` exposes these callables:

```python
self.assertTrue(callable(modal_train.sft))
self.assertTrue(callable(modal_train.eval_validity))
self.assertTrue(callable(modal_train.rl_smoke))
```

Run:

```bash
PYTHONPATH=src python3 -m unittest tests/test_modal_readiness.py -q
```

Expected: fail until the entrypoints exist.

- [ ] **Step 2: Implement only thin entrypoint shells**

Each shell must:

- use the existing `app`, `image`, `volumes`, and `secrets`;
- call the future local trainer/eval/RL functions;
- write a local JSON report to `/runs/<run_id>/reports/` before or alongside W&B logging;
- reject online mode without `WANDB_API_KEY`;
- sample RL actions only from NLE action IDs discovered by the active environment.

- [ ] **Step 3: Run static gates**

```bash
PYTHONPATH=src python3 -m unittest tests/test_modal_readiness.py -q
git diff --check
```

Expected: tests pass and no whitespace errors.

## Failure Gates

Run these before declaring Modal readiness complete:

```bash
PYTHONPATH=src python3 -m unittest tests/test_modal_readiness.py -q
python3 -m compileall src tests
git diff --check
rg -n "hf_[A-Za-z0-9]{20,}|wandb_[A-Za-z0-9]{20,}|WANDB_API_KEY=[A-Za-z0-9_-]{20,}|HF_TOKEN=[A-Za-z0-9_-]{20,}|WATCH_AUTH_TOKEN=[A-Za-z0-9_-]{20,}" .
```

Expected:

- Unit and compile gates pass.
- `git diff --check` prints nothing.
- Secret scan prints nothing.
- No raw NLD/NLE data, checkpoints, ttyrecs, replay media, W&B credentials, or Modal cache state are tracked.

## Handoff Rules

- Do not run expensive GPU training from this lane.
- Do not add BALROG to `modal-train`; use a separate optional evaluator environment later.
- Do not use BALROG file-based secrets.
- Do not claim NetHack competence from readiness or two-episode smoke output.
- Keep W&B mandatory and mirrored by local JSON reports.
