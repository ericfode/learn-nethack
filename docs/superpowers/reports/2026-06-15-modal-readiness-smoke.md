# Modal Readiness Smoke - 2026-06-15

## Offline Cloud Smoke

- Command: `WANDB_MODE=offline modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-offline-20260615-vantage4`
- Status: pass
- Modal app run: `https://modal.com/apps/ericfode/main/ap-ohBBboLzK0mTiyiFUzBxH8`
- Report: `/runs/modal-readiness-offline-20260615-vantage4/reports/modal_readiness_report.json`
- Downloaded verification copy: `/private/tmp/modal_readiness_report_vantage4.json`
- W&B offline run: `/runs/modal-readiness-offline-20260615-vantage4/wandb/wandb/offline-run-20260615_231120-sg3nqywe`
- Execution evidence: `execution.backend` was `modal_cloud` with function call `fc-01KV6RVC3ATAFATX894FFG2D1S`, input `in-01KV6RVC3EKRK8WG87APGZXRPB:1781565075567-0`, and task `ta-01KV6RVCY80B0GQT5H7E3P66JA`.
- Notes: offline W&B was explicit; the training secret was not required for this smoke; no secrets were written to git.

## Pre-Fix Offline Blocker

- Command attempted before the offline fix: `WANDB_MODE=offline modal run src/learn_nethack/modal_train.py::readiness --run-id modal-readiness-offline-20260615-vantage`
- Status: failed before function execution
- Cause: `learn-nethack-training-secrets` did not exist in Modal environment `main`.
- Resolution: explicit offline runs now omit secrets and inject `WANDB_MODE=offline`.
  Credentialed online runs now use the existing Modal secrets `hf-token` and
  `wandb-secret`; do not recreate the retired `learn-nethack-training-secrets`
  secret.
