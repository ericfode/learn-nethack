"""Static project-status dashboard for the NetHack Gemma training goal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

from learn_nethack.compare_watch import FITNESS_OBJECTIVE_VERSION
from learn_nethack.modal_config import MODAL_APP_NAME
from learn_nethack.wandb_logging import build_wandb_visibility_report


DASHBOARD_SCHEMA_VERSION = "learn-nethack.status-dashboard.v1"
DEFAULT_FULL_BUILD_RUN_ID = (
    "full-archive-pseudo-dynamics-build-full-feedback-context6-20260618-01"
)
DEFAULT_BASELINE_EVAL_RUN_ID = (
    "full-archive-pseudo-dynamics-sft-full-feedback-context6-baseline-gen64-seq64-"
    "eval-20260618-01"
)


Runner = Callable[[Sequence[str]], Mapping[str, Any]]


@dataclass(frozen=True)
class DashboardWriteResult:
    """Paths written by the dashboard generator."""

    out_dir: Path
    snapshot_path: Path
    html_path: Path
    snapshot: dict[str, Any]


def write_status_dashboard(
    *,
    repo_root: Path,
    out_dir: Path,
    build_run_id: str = DEFAULT_FULL_BUILD_RUN_ID,
    baseline_eval_run_id: str = DEFAULT_BASELINE_EVAL_RUN_ID,
    refresh_modal_apps: bool = True,
    modal_runner: Runner | None = None,
) -> DashboardWriteResult:
    """Write a source-backed HTML dashboard and compact JSON snapshot."""
    snapshot = build_dashboard_snapshot(
        repo_root=repo_root,
        build_run_id=build_run_id,
        baseline_eval_run_id=baseline_eval_run_id,
        refresh_modal_apps=refresh_modal_apps,
        modal_runner=modal_runner,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "dashboard.json"
    html_path = out_dir / "index.html"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(
        render_dashboard_html(snapshot=snapshot, html_path=html_path),
        encoding="utf-8",
    )
    return DashboardWriteResult(
        out_dir=out_dir,
        snapshot_path=snapshot_path,
        html_path=html_path,
        snapshot=snapshot,
    )


def build_dashboard_snapshot(
    *,
    repo_root: Path,
    build_run_id: str = DEFAULT_FULL_BUILD_RUN_ID,
    baseline_eval_run_id: str = DEFAULT_BASELINE_EVAL_RUN_ID,
    refresh_modal_apps: bool = True,
    modal_runner: Runner | None = None,
) -> dict[str, Any]:
    """Collect the current project status from local artifacts and light probes."""
    root = repo_root.resolve()
    artifacts = root / "artifacts"
    full_build = _read_json(
        artifacts / build_run_id / "full_build_status.json",
        fallback={"build_run_id": build_run_id, "missing": True},
    )
    baseline_eval = _read_json(
        artifacts / baseline_eval_run_id / "eval_status.json",
        fallback={"eval_run_id": baseline_eval_run_id, "missing": True},
    )
    wandb_status = build_wandb_visibility_report(root=root)
    modal_apps = (
        _load_modal_apps(runner=modal_runner)
        if refresh_modal_apps
        else {"status": "skipped", "apps": []}
    )
    shard_summary = _collect_shard_summary(artifacts / "modal-shards")
    watch_demos = _collect_watch_demos(artifacts)
    score_reports = _collect_score_reports(artifacts)
    proof_gates = _collect_proof_gates(artifacts)
    world_model_proof = _collect_world_model_proof(artifacts)
    training_reports = _collect_training_reports(artifacts)
    goal_status = _goal_status(
        full_build=full_build,
        baseline_eval=baseline_eval,
        proof_gates=proof_gates,
        modal_apps=modal_apps,
    )
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": str(root),
        "objective": (
            "Produce a Gemma 4 checkpoint that beats frozen base and current-20k "
            "on corrected policy/dynamics evidence and paired live NetHack play. "
            "Scale only after the corrected-20k proof gate passes."
        ),
        "fitness_objective_version": FITNESS_OBJECTIVE_VERSION,
        "goal_status": goal_status,
        "full_build": full_build,
        "baseline_eval": baseline_eval,
        "wandb": wandb_status,
        "modal_apps": modal_apps,
        "shards": shard_summary,
        "watch_demos": watch_demos,
        "score_reports": score_reports,
        "proof_gates": proof_gates,
        "world_model_proof": world_model_proof,
        "training_reports": training_reports,
        "todo": _build_todo_items(
            full_build=full_build,
            baseline_eval=baseline_eval,
            wandb_status=wandb_status,
            proof_gates=proof_gates,
        ),
        "sources": _source_manifest(
            artifacts=artifacts,
            build_run_id=build_run_id,
            baseline_eval_run_id=baseline_eval_run_id,
        ),
    }


def render_dashboard_html(*, snapshot: Mapping[str, Any], html_path: Path) -> str:
    """Render a self-contained HTML dashboard from a collected snapshot."""
    full_build = dict(snapshot.get("full_build") or {})
    wandb = dict(snapshot.get("wandb") or {})
    shards = dict(snapshot.get("shards") or {})
    goal_status = dict(snapshot.get("goal_status") or {})
    demos = list(snapshot.get("watch_demos") or [])
    score_reports = list(snapshot.get("score_reports") or [])
    proof_gates = list(snapshot.get("proof_gates") or [])
    world_model_proof = dict(snapshot.get("world_model_proof") or {})
    modal_apps = dict(snapshot.get("modal_apps") or {})
    todo = list(snapshot.get("todo") or [])
    latest_score = _best_corrected_policy_report(score_reports)
    latest_exact = dict(latest_score.get("exact_match_rate") or {})
    latest_gate = proof_gates[0] if proof_gates else {}
    training_reports = list(snapshot.get("training_reports") or [])
    latest_training = training_reports[0] if training_reports else {}
    online_wandb_recorded = any(report.get("wandb_url") for report in training_reports)

    selected_demo = _first_demo_with_frame(demos)
    demo_src = (
        _relative_url(html_path.parent, Path(selected_demo["demo_path"]))
        if selected_demo
        else ""
    )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Learn NetHack Training Dashboard</title>",
            "<style>",
            _DASHBOARD_CSS,
            "</style>",
            "</head>",
            "<body>",
            '<main class="shell">',
            '<section class="header">',
            "<div>",
            '<p class="eyebrow">Learn NetHack</p>',
            "<h1>Training Goal Dashboard</h1>",
            f"<p>{_esc(snapshot.get('objective'))}</p>",
            f'<p class="decision-reason">{_esc(goal_status.get("reason"))}</p>',
            "</div>",
            f'<span class="status-pill status-{_esc(goal_status.get("tone", "warn"))}">{_esc(goal_status.get("label", "Unknown"))}</span>',
            "</section>",
            '<section class="kpi-grid">',
            _kpi_card(
                "Corrected 20k policy",
                _format_rate(latest_exact.get("trained")),
                f"baseline {_format_rate(latest_exact.get('baseline'))}; offline exact match",
                tone="good" if latest_exact.get("improved") else "warn",
            ),
            _kpi_card(
                "Live proof gate",
                "Passed" if latest_gate.get("passed") else "Failed",
                f"{_format_int(latest_gate.get('failed_count'))} failed requirements",
                tone="good" if latest_gate.get("passed") else "bad",
            ),
            _kpi_card(
                "Latest training",
                str(latest_training.get("status") or "Waiting").title(),
                f"loss {_format_float(latest_training.get('train_loss'))}",
                tone="good" if latest_training.get("status") == "completed" else "warn",
            ),
            _kpi_card(
                "W&B evidence",
                "Online runs recorded"
                if online_wandb_recorded
                else "No online run found",
                f"{_format_int(wandb.get('offline_run_count'))} local offline runs",
                tone="good" if online_wandb_recorded else "bad",
            ),
            "</section>",
            '<section class="two-col">',
            '<div class="panel">',
            "<h2>Pipeline State</h2>",
            _pipeline_html(snapshot),
            "</div>",
            '<div class="panel">',
            "<h2>What Is Left</h2>",
            _todo_html(todo),
            "</div>",
            "</section>",
            '<section class="panel">',
            "<h2>Full Dataset Build</h2>",
            _build_detail_html(full_build, shards),
            "</section>",
            '<section class="panel">',
            "<h2>Evaluation And Proof</h2>",
            _score_reports_html(score_reports),
            _proof_gate_html(proof_gates),
            "</section>",
            '<section class="panel">',
            "<h2>Local Decoder Proof</h2>",
            _world_model_proof_html(world_model_proof, html_path),
            "</section>",
            '<section class="panel">',
            "<h2>Live Demos</h2>",
            '<p class="muted">These are watchable rollout artifacts already written by the project. Open a demo or switch the embedded viewer.</p>',
            _demo_buttons_html(demos, html_path),
            _demo_frame_html(demo_src),
            _watch_table_html(demos, html_path),
            "</section>",
            '<section class="two-col">',
            '<div class="panel">',
            "<h2>Modal</h2>",
            _modal_apps_html(modal_apps),
            "</div>",
            '<div class="panel">',
            "<h2>Sources</h2>",
            _sources_html(snapshot.get("sources") or []),
            "</div>",
            "</section>",
            "</main>",
            "<script>",
            _DASHBOARD_JS,
            "</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _goal_status(
    *,
    full_build: Mapping[str, Any],
    baseline_eval: Mapping[str, Any],
    proof_gates: Sequence[Mapping[str, Any]],
    modal_apps: Mapping[str, Any],
) -> dict[str, str]:
    latest_gate = proof_gates[0] if proof_gates else {}
    if latest_gate:
        if latest_gate.get("passed") is True:
            return {
                "label": "Improvement proved",
                "tone": "good",
                "reason": "Latest corrected proof gate passed.",
            }
        failed_names = {
            str(item.get("name")) for item in latest_gate.get("failed") or []
        }
        live_control_failed = bool(
            failed_names
            & {
                "watch_current_action_repeat_rate_ceiling",
                "watch_current_zero_progress_episode_ceiling",
                "watch_current_score_or_depth_progress",
                "watch_action_repeat_rate",
                "watch_score_or_depth_progress",
            }
        )
        return {
            "label": (
                "Corrected 20k rejected: live control failed"
                if live_control_failed
                else "Corrected 20k proof failed"
            ),
            "tone": "bad",
            "reason": (
                "Offline imitation improved, but paired live rollouts failed "
                "state-contingent control and made no score, reward, or depth "
                "progress. Full-corpus scaling is withheld."
                if live_control_failed
                else "The latest corrected proof gate did not pass."
            ),
            "build_activity": "withheld_by_proof_gate",
        }
    if not full_build.get("train_ready"):
        build_activity = _full_build_activity(
            full_build=full_build,
            modal_apps=modal_apps,
        )
        if build_activity == "running":
            return {
                "label": "Goal active: full build still running",
                "tone": "warn",
                "reason": "An active Modal build task exists; completion markers are missing.",
                "build_activity": build_activity,
            }
        if build_activity == "stalled":
            return {
                "label": "Goal blocked: full build stalled",
                "tone": "bad",
                "reason": (
                    "No active Modal task matches this build and completion markers "
                    "are missing."
                ),
                "build_activity": build_activity,
            }
        return {
            "label": "Goal active: full build incomplete",
            "tone": "warn",
            "reason": "The full dataset is not train-ready; Modal activity is unknown.",
            "build_activity": build_activity,
        }
    if not baseline_eval.get("eval_ready"):
        return {
            "label": "Goal active: baseline eval incomplete",
            "tone": "warn",
            "reason": "Baseline metrics are needed before proof.",
        }
    latest_gate = proof_gates[0] if proof_gates else {}
    if latest_gate.get("passed") is True:
        return {
            "label": "Improvement proved",
            "tone": "good",
            "reason": "Latest proof gate passed.",
        }
    return {
        "label": "Goal active: proof not passed",
        "tone": "bad",
        "reason": "No current proof gate proves full-dataset improvement.",
    }


def _full_build_activity(
    *,
    full_build: Mapping[str, Any],
    modal_apps: Mapping[str, Any],
) -> str:
    if modal_apps.get("status") != "ok":
        return "unknown"
    build_run_id = str(full_build.get("build_run_id") or "")
    for app in modal_apps.get("apps") or []:
        description = str(_first_present(app, "description", "Description") or "")
        describes_build = description == MODAL_APP_NAME or (
            bool(build_run_id) and build_run_id in description
        )
        if describes_build and _has_active_tasks(_first_present(app, "tasks", "Tasks")):
            return "running"
    return "stalled"


def _has_active_tasks(value: Any) -> bool:
    try:
        return int(str(value).strip()) > 0
    except (TypeError, ValueError):
        return False


def _build_todo_items(
    *,
    full_build: Mapping[str, Any],
    baseline_eval: Mapping[str, Any],
    wandb_status: Mapping[str, Any],
    proof_gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    latest_gate = proof_gates[0] if proof_gates else {}
    if latest_gate and latest_gate.get("passed") is not True:
        return [
            {
                "label": "Corrected 20k offline policy gate",
                "status": "done",
                "detail": "Assistant-only true-keypress SFT improved held-out policy metrics without offline collapse.",
            },
            {
                "label": "Paired live smoke gate",
                "status": "failed",
                "detail": "Single-frame collapsed to SEARCH; growing-context replayed the same action sequence across roles; both made zero progress.",
            },
            {
                "label": "Correct live distribution shift",
                "status": "active",
                "detail": "Add on-policy recovery states, action-balance controls, and anti-repetition training before more rollout scale.",
            },
            {
                "label": "Train and gate corrected dynamics",
                "status": "pending",
                "detail": "Must beat copy-current and deterministic next-1/5/10 changed-state baselines.",
            },
            {
                "label": "Run 16-role live proof",
                "status": "blocked",
                "detail": "Blocked until a paired smoke shows absolute progress without collapse.",
            },
            {
                "label": "Scale to full corpus",
                "status": "blocked",
                "detail": "Corrected-20k promotion gate failed; full-corpus training is not authorized.",
            },
        ]
    items = [
        {
            "label": "Finish full-dataset SFT build",
            "status": "done" if full_build.get("train_ready") else "active",
            "detail": _join_list(full_build.get("missing_markers"), "missing"),
        },
        {
            "label": "Record baseline next-1/5/10 eval",
            "status": "done" if baseline_eval.get("eval_ready") else "active",
            "detail": _join_list(baseline_eval.get("missing_markers"), "missing"),
        },
        {
            "label": "Run full-dataset SFT on Modal",
            "status": "pending" if not full_build.get("train_ready") else "active",
            "detail": "Requires completed manifest/rejection/build reports.",
        },
        {
            "label": "Evaluate trained checkpoint",
            "status": "pending",
            "detail": "Run matched policy and next-frame eval against baseline.",
        },
        {
            "label": "Run 16-seed live watch proof",
            "status": "pending",
            "detail": (
                f"Must use {FITNESS_OBJECTIVE_VERSION} and NetHackPairedChallenge-v0."
            ),
        },
        {
            "label": "Pass proof gate",
            "status": "done"
            if proof_gates and proof_gates[0].get("passed")
            else "pending",
            "detail": "Needs next-frame gains plus score/reward/depth progress without damage/stuck regressions.",
        },
        {
            "label": "W&B visibility",
            "status": "done" if wandb_status.get("api_key_configured") else "active",
            "detail": "Local shell still needs WANDB_API_KEY or wandb login for sync.",
        },
    ]
    return items


def _collect_shard_summary(shard_dir: Path) -> dict[str, Any]:
    reports = sorted(shard_dir.glob("nld-nao-shard-*.db-report.json"))
    total_games = 0
    total_ttyrecs = 0
    valid_reports = 0
    for path in reports:
        payload = _read_json(path, fallback={})
        if not payload:
            continue
        valid_reports += 1
        total_games += int(payload.get("selected_game_count") or 0)
        total_ttyrecs += int(payload.get("selected_ttyrec_count") or 0)
    return {
        "report_count": valid_reports,
        "total_games": total_games,
        "total_ttyrecs": total_ttyrecs,
        "source_dir": str(shard_dir),
    }


def _collect_watch_demos(artifacts: Path) -> list[dict[str, Any]]:
    reports = list(artifacts.glob("**/report.json")) + list(
        artifacts.glob("**/sweep_report.json")
    )
    demos: list[dict[str, Any]] = []
    for report_path in reports:
        payload = _read_json(report_path, fallback={})
        if payload.get("schema_version") == "learn-nethack.local-world-model-proof.v1":
            index_path = report_path.parent / "watch" / "index.html"
            if index_path.exists():
                demos.append(
                    {
                        "run_id": payload.get("run_id") or report_path.parent.name,
                        "kind": "world-model",
                        "report_path": str(report_path),
                        "demo_path": str(index_path),
                        "demo_label": f"World model: {report_path.parent.name}",
                        "mtime": report_path.stat().st_mtime,
                        "current": {},
                        "baseline": {},
                        "deltas": {},
                    }
                )
            continue
        if not isinstance(payload.get("rollout_metrics"), Mapping):
            continue
        run_id = str(payload.get("run_id") or report_path.parent.name)
        metrics = dict(payload.get("rollout_metrics") or {})
        index_path = report_path.parent / "index.html"
        demo_paths: list[Path] = []
        if index_path.exists():
            demo_paths.append(index_path)
        demo_paths.extend(sorted(report_path.parent.glob("seed-*/index.html")))
        if not demo_paths:
            demo_paths.append(report_path)
        for demo_path in demo_paths[:3]:
            demos.append(
                {
                    "run_id": run_id,
                    "kind": (
                        "sweep" if report_path.name == "sweep_report.json" else "single"
                    ),
                    "report_path": str(report_path),
                    "demo_path": str(demo_path),
                    "demo_label": (
                        f"{demo_path.parent.parent.name} / {demo_path.parent.name}"
                        if demo_path.parent.name.startswith("seed-")
                        else run_id
                    ),
                    "mtime": report_path.stat().st_mtime,
                    "current": _compact_rollout_metrics(metrics.get("current")),
                    "baseline": _compact_rollout_metrics(metrics.get("baseline")),
                    "deltas": _compact_rollout_metrics(metrics.get("deltas")),
                }
            )
    demos.sort(
        key=lambda row: (float(row.get("mtime") or 0.0), row["run_id"]), reverse=True
    )
    unique: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for demo in demos:
        demo_path = str(demo.get("demo_path") or "")
        if demo_path in seen_paths:
            continue
        seen_paths.add(demo_path)
        unique.append(demo)
    return unique[:12]


def _compact_rollout_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        return {}
    keys = (
        "fitness_objective_version",
        "fitness_score",
        "cumulative_reward",
        "score_delta",
        "depth_delta",
        "depth_max",
        "hp_damage_observed",
        "wall_message_rate",
        "bad_message_rate",
        "action_repeat_rate",
        "non_advancing_step_rate",
        "menu_or_prompt_step_rate",
        "dirty_live_progress_event_count",
        "zero_progress_episode",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _collect_score_reports(artifacts: Path) -> list[dict[str, Any]]:
    reports = sorted(artifacts.glob("**/score_to_beat*.json"))
    rows: list[dict[str, Any]] = []
    for path in reports:
        payload = _read_json(path, fallback={})
        metrics = dict(payload.get("metrics") or {})
        if not metrics:
            continue
        rows.append(
            {
                "path": str(path),
                "run": path.parent.name,
                "mtime": path.stat().st_mtime,
                "verdict": payload.get("verdict"),
                "baseline_run_id": payload.get("baseline_run_id"),
                "trained_run_id": payload.get("trained_run_id"),
                "exact_match_rate": metrics.get("exact_match_rate"),
                "next_1": metrics.get("next_1_frame_sequence_char_accuracy"),
                "next_5": metrics.get("next_5_frame_sequence_char_accuracy"),
                "next_10": metrics.get("next_10_frame_sequence_char_accuracy"),
            }
        )
    rows.sort(
        key=lambda row: (float(row.get("mtime") or 0.0), row["run"]), reverse=True
    )
    return rows[:8]


def _collect_proof_gates(artifacts: Path) -> list[dict[str, Any]]:
    reports = sorted(artifacts.glob("**/training_proof_gate*.json"))
    rows: list[dict[str, Any]] = []
    for path in reports:
        payload = _read_json(path, fallback={})
        if payload.get("schema_version") != "learn-nethack.training-proof-gate.v1":
            continue
        failed = [
            {
                "name": requirement.get("name"),
                "reason": requirement.get("reason"),
            }
            for requirement in payload.get("requirements", [])
            if requirement.get("status") == "failed"
        ]
        rows.append(
            {
                "path": str(path),
                "run": path.parent.name,
                "mtime": path.stat().st_mtime,
                "passed": payload.get("passed"),
                "verdict": payload.get("verdict"),
                "failed_count": len(failed),
                "failed": failed[:8],
            }
        )
    rows.sort(
        key=lambda row: (float(row.get("mtime") or 0.0), row["path"]), reverse=True
    )
    return rows[:6]


def _collect_world_model_proof(artifacts: Path) -> dict[str, Any]:
    candidates = sorted(
        artifacts.glob("**/*world-model*aggregate*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        payload = _read_json(path, fallback={})
        if (
            payload.get("schema_version")
            != "learn-nethack.local-world-model-aggregate.v1"
        ):
            continue
        return {**payload, "path": str(path), "mtime": path.stat().st_mtime}
    return {}


def _collect_training_reports(artifacts: Path) -> list[dict[str, Any]]:
    paths = list(artifacts.glob("**/sft_train_report.json")) + list(
        artifacts.glob("**/sft_train_existing_report.json")
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path, fallback={})
        training_plan = dict(payload.get("training_plan") or {})
        train_metrics = dict(payload.get("train_metrics") or {})
        rows.append(
            {
                "path": str(path),
                "run_id": payload.get("run_id") or path.parent.name,
                "status": payload.get("status"),
                "model_name": training_plan.get("model_name"),
                "training_objective": training_plan.get("training_objective"),
                "max_steps": training_plan.get("max_steps"),
                "train_loss": train_metrics.get("train_loss"),
                "wandb_url": (payload.get("wandb") or {}).get("run_url"),
                "mtime": path.stat().st_mtime,
            }
        )
    rows.sort(
        key=lambda row: (float(row.get("mtime") or 0.0), row["run_id"]), reverse=True
    )
    return rows[:6]


def _load_modal_apps(*, runner: Runner | None) -> dict[str, Any]:
    effective_runner = runner or _run_subprocess
    result = dict(effective_runner(["modal", "app", "list", "--json"]))
    if result.get("returncode") != 0:
        return {
            "status": "error",
            "apps": [],
            "stderr": result.get("stderr"),
            "command": "modal app list --json",
        }
    try:
        apps = json.loads(str(result.get("stdout") or "[]"))
    except json.JSONDecodeError as exc:
        return {
            "status": "parse_error",
            "apps": [],
            "stderr": str(exc),
            "command": "modal app list --json",
        }
    return {"status": "ok", "apps": apps, "command": "modal app list --json"}


def _source_manifest(
    *, artifacts: Path, build_run_id: str, baseline_eval_run_id: str
) -> list[dict[str, str]]:
    return [
        {
            "label": "Full build status",
            "path": str(artifacts / build_run_id / "full_build_status.json"),
        },
        {
            "label": "Baseline eval status",
            "path": str(artifacts / baseline_eval_run_id / "eval_status.json"),
        },
        {"label": "Watch reports", "path": str(artifacts / "watch")},
        {"label": "Training/eval artifacts", "path": str(artifacts)},
        {"label": "W&B visibility", "path": "computed from local wandb dirs/env"},
        {"label": "Modal apps", "path": "modal app list --json"},
    ]


def _pipeline_html(snapshot: Mapping[str, Any]) -> str:
    full_build = dict(snapshot.get("full_build") or {})
    baseline_eval = dict(snapshot.get("baseline_eval") or {})
    proof_gates = list(snapshot.get("proof_gates") or [])
    proof_passed = bool(proof_gates and proof_gates[0].get("passed"))
    steps = [
        ("Archive shards staged", "done", "31 shard reports in local artifacts"),
        (
            "Full SFT build",
            "done" if full_build.get("train_ready") else "active",
            _join_list(full_build.get("missing_markers"), "missing"),
        ),
        (
            "Baseline eval",
            "done" if baseline_eval.get("eval_ready") else "active",
            _join_list(baseline_eval.get("missing_markers"), "missing"),
        ),
        (
            "Full SFT training",
            "pending" if not full_build.get("train_ready") else "active",
            "Waiting for train-ready dataset markers",
        ),
        ("Trained eval", "pending", "Needs checkpoint from full SFT run"),
        (
            "Proof gate",
            "done" if proof_passed else "pending",
            f"Current objective: {FITNESS_OBJECTIVE_VERSION}",
        ),
    ]
    return (
        '<div class="timeline">'
        + "".join(
            f'<div class="step step-{_esc(status)}"><b>{_esc(label)}</b><span>{_esc(detail)}</span></div>'
            for label, status, detail in steps
        )
        + "</div>"
    )


def _todo_html(todo: Sequence[Mapping[str, Any]]) -> str:
    if not todo:
        return '<p class="muted">No todo items found.</p>'
    return (
        '<ul class="todo">'
        + "".join(
            f'<li class="todo-{_esc(item.get("status", "pending"))}"><b>{_esc(item.get("label"))}</b><span>{_esc(item.get("detail"))}</span></li>'
            for item in todo
        )
        + "</ul>"
    )


def _build_detail_html(full_build: Mapping[str, Any], shards: Mapping[str, Any]) -> str:
    progress = dict(full_build.get("progress") or {})
    latest = dict(progress.get("latest") or {})
    accepted = int(latest.get("accepted_policy_rows") or 0)
    rejected = int(latest.get("rejected_rows") or 0)
    processed = int(latest.get("processed_transitions") or 0)
    total_seen = accepted + rejected
    acceptance = accepted / total_seen if total_seen else 0.0
    return "\n".join(
        [
            '<div class="metric-row">',
            _small_metric("Processed transitions", _format_int(processed)),
            _small_metric("Accepted policy rows", _format_int(accepted)),
            _small_metric(
                "Accepted next-frame rows",
                _format_int(latest.get("accepted_next_frame_rows")),
            ),
            _small_metric("Rejected rows", _format_int(rejected)),
            _small_metric("Restarts", _format_int(progress.get("restart_count"))),
            "</div>",
            _progress_bar(
                acceptance,
                f"Acceptance among seen transitions: {_format_pct(acceptance)}",
            ),
            '<div class="metric-row">',
            _small_metric("Shard reports", _format_int(shards.get("report_count"))),
            _small_metric("Shard games", _format_int(shards.get("total_games"))),
            _small_metric("Shard ttyrecs", _format_int(shards.get("total_ttyrecs"))),
            _small_metric(
                "Train ready", "yes" if full_build.get("train_ready") else "no"
            ),
            "</div>",
            f'<p class="muted">Next action: {_esc(full_build.get("next_action"))}</p>',
        ]
    )


def _score_reports_html(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">No score-to-beat reports found.</p>'
    body = "".join(
        "<tr>"
        f"<td>{_esc(row.get('run'))}</td>"
        f"<td>{_esc(row.get('verdict'))}</td>"
        f"<td>{_metric_delta(row.get('exact_match_rate'))}</td>"
        f"<td>{_metric_delta(row.get('next_1'))}</td>"
        f"<td>{_metric_delta(row.get('next_5'))}</td>"
        f"<td>{_metric_delta(row.get('next_10'))}</td>"
        "</tr>"
        for row in rows[:5]
    )
    return (
        "<h3>Recent Score-To-Beat Reports</h3><table><thead><tr>"
        "<th>Run</th><th>Verdict</th><th>Action exact</th><th>Next-1</th><th>Next-5</th><th>Next-10</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _proof_gate_html(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">No proof-gate reports found.</p>'
    latest = rows[0]
    failures = latest.get("failed") or []
    failure_html = "".join(
        f"<li><b>{_esc(item.get('name'))}</b>: {_esc(item.get('reason'))}</li>"
        for item in failures
    )
    return "\n".join(
        [
            '<div class="proof-box">',
            f"<h3>Latest Proof Gate: {_esc(latest.get('verdict'))}</h3>",
            f"<p>Passed: {_esc(latest.get('passed'))}. Failed requirements: {_format_int(latest.get('failed_count'))}.</p>",
            f"<ul>{failure_html}</ul>" if failures else "",
            f'<p class="muted">{_esc(latest.get("path"))}</p>',
            "</div>",
        ]
    )


def _world_model_proof_html(
    proof: Mapping[str, Any],
    html_path: Path,
) -> str:
    if not proof:
        return '<p class="muted">No local decoder comparison found.</p>'
    variants = dict(proof.get("variants") or {})
    deterministic = dict(variants.get("deterministic") or {})
    diffusion = dict(variants.get("diffusion") or {})
    deltas = dict(proof.get("diffusion_minus_deterministic") or {})
    rows = []
    for label, key in (
        ("One-step changed F1", "one_step_changed_f1"),
        ("Next-1 changed F1", "next_1_changed_f1"),
        ("Next-5 changed F1", "next_5_changed_f1"),
        ("Next-10 changed F1", "next_10_changed_f1"),
        ("Action MRR", "action_mrr"),
        ("One-step char accuracy", "one_step_char_accuracy"),
    ):
        rows.append(
            "<tr>"
            f"<td>{_esc(label)}</td>"
            f"<td>{_aggregate_mean_std(deterministic.get(key))}</td>"
            f"<td>{_aggregate_mean_std(diffusion.get(key))}</td>"
            f"<td>{_aggregate_delta(deltas.get(key))}</td>"
            "</tr>"
        )
    aggregate_path = Path(str(proof.get("path")))
    aggregate_url = _relative_url(html_path.parent, aggregate_path)
    failures = dict(proof.get("failure_summary") or {})
    verdict = str(proof.get("verdict") or "unknown")
    return "".join(
        [
            '<div class="proof-box">',
            f"<h3>Diffusion replacement: {_esc(verdict)}</h3>",
            f"<p>{_format_int(proof.get('supported_run_count'))} of {_format_int(proof.get('run_count'))} training seeds passed the pre-registered gate. "
            f"Diffusion lost action MRR in {_format_int(failures.get('action_mrr_nonpositive_delta_runs'))} runs.</p>",
            "<table><thead><tr><th>Metric</th><th>Deterministic</th><th>Diffusion</th><th>Delta</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody></table>",
            f'<p class="muted">Matched parameters: {_format_int(proof.get("matched_parameter_count"))}. '
            f"Diffusion next-10 F1 range: {_num(failures.get('diffusion_next_10_f1_range'))}. "
            f'<a href="{_esc(aggregate_url)}">Aggregate JSON</a></p>',
            "</div>",
        ]
    )


def _aggregate_mean_std(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "n/a"
    return f"{_num(value.get('mean'))} +/- {_num(value.get('sample_std'))}"


def _aggregate_delta(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "n/a"
    return _num(value.get("mean"), signed=True)


def _demo_buttons_html(demos: Sequence[Mapping[str, Any]], html_path: Path) -> str:
    if not demos:
        return '<p class="muted">No watch demos found.</p>'
    buttons = []
    for demo in demos[:6]:
        demo_path = Path(str(demo.get("demo_path")))
        url = _relative_url(html_path.parent, demo_path)
        buttons.append(
            f'<button class="demo-button" data-demo-src="{_esc(url)}">{_esc(demo.get("demo_label"))}</button>'
        )
    return '<div class="demo-buttons">' + "".join(buttons) + "</div>"


def _demo_frame_html(src: str) -> str:
    if not src:
        return ""
    return f'<iframe id="demo-frame" title="Watch demo" src="{_esc(src)}"></iframe>'


def _watch_table_html(demos: Sequence[Mapping[str, Any]], html_path: Path) -> str:
    if not demos:
        return ""
    body = "".join(
        "<tr>"
        f'<td><a href="{_esc(_relative_url(html_path.parent, Path(str(row.get("demo_path")))))}">{_esc(row.get("demo_label"))}</a></td>'
        f"<td>{_esc(row.get('kind'))}</td>"
        f"<td>{_num((row.get('current') or {}).get('cumulative_reward'))}</td>"
        f"<td>{_num((row.get('current') or {}).get('score_delta'))}</td>"
        f"<td>{_num((row.get('current') or {}).get('hp_damage_observed'))}</td>"
        f"<td>{_num((row.get('current') or {}).get('wall_message_rate'))}</td>"
        f"<td>{_num((row.get('current') or {}).get('action_repeat_rate'))}</td>"
        f"<td>{_esc((row.get('current') or {}).get('fitness_objective_version'))}</td>"
        "</tr>"
        for row in demos[:10]
    )
    return (
        "<table><thead><tr><th>Demo</th><th>Type</th><th>Reward</th><th>Score</th>"
        "<th>HP damage</th><th>Wall rate</th><th>Repeat rate</th><th>Objective</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _modal_apps_html(modal_apps: Mapping[str, Any]) -> str:
    apps = list(modal_apps.get("apps") or [])
    if not apps:
        return f'<p class="muted">Modal app list unavailable: {_esc(modal_apps.get("status"))}</p>'
    body = "".join(
        "<tr>"
        f"<td>{_esc(_first_present(app, 'app_id', 'App ID'))}</td>"
        f"<td>{_esc(_first_present(app, 'description', 'Description'))}</td>"
        f"<td>{_esc(_first_present(app, 'state', 'State'))}</td>"
        f"<td>{_esc(_first_present(app, 'tasks', 'Tasks'))}</td>"
        "</tr>"
        for app in apps
    )
    return f"<table><thead><tr><th>App</th><th>Description</th><th>State</th><th>Tasks</th></tr></thead><tbody>{body}</tbody></table>"


def _sources_html(sources: Sequence[Mapping[str, Any]]) -> str:
    return (
        '<ul class="sources">'
        + "".join(
            f"<li><b>{_esc(source.get('label'))}</b><span>{_esc(source.get('path'))}</span></li>"
            for source in sources
        )
        + "</ul>"
    )


def _kpi_card(title: str, value: str, detail: str, *, tone: str) -> str:
    return (
        f'<div class="kpi kpi-{_esc(tone)}"><span>{_esc(title)}</span>'
        f"<b>{_esc(value)}</b><p>{_esc(detail)}</p></div>"
    )


def _small_metric(label: str, value: str) -> str:
    return f'<div class="small-metric"><span>{_esc(label)}</span><b>{_esc(value)}</b></div>'


def _progress_bar(value: float, label: str) -> str:
    percent = max(0.0, min(1.0, value)) * 100.0
    return (
        f'<div class="bar-label">{_esc(label)}</div>'
        f'<div class="bar"><span style="width:{percent:.1f}%"></span></div>'
    )


def _first_demo_with_frame(
    demos: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    for demo in demos:
        if str(demo.get("demo_path") or "").endswith(".html"):
            return demo
    return demos[0] if demos else None


def _read_json(path: Path, *, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _run_subprocess(command: Sequence[str]) -> Mapping[str, Any]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _relative_url(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve())


def _format_build_progress(
    latest: Mapping[str, Any], full_build: Mapping[str, Any]
) -> str:
    processed = _format_int(latest.get("processed_transitions"))
    missing = _join_list(full_build.get("missing_markers"), "missing")
    return f"{processed} transitions; {missing}"


def _join_list(value: Any, empty_label: str) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [str(item) for item in value if str(item)]
        if items:
            return f"{', '.join(items)} {empty_label}"
    return f"no {empty_label}"


def _metric_delta(metric: Any) -> str:
    if not isinstance(metric, Mapping):
        return "n/a"
    baseline = _num(metric.get("baseline"))
    trained = _num(metric.get("trained"))
    delta = _num(metric.get("delta"), signed=True)
    marker = (
        "up"
        if metric.get("improved")
        else "down"
        if metric.get("regressed")
        else "flat"
    )
    return f'<span class="metric-{marker}">{baseline} -> {trained} ({delta})</span>'


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _best_corrected_policy_report(
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    corrected = [
        report
        for report in reports
        if str(report.get("run") or "").startswith("corrected20k-")
    ]
    candidates = corrected or list(reports)
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda report: float(
            (report.get("exact_match_rate") or {}).get("trained") or 0.0
        ),
    )


def _format_rate(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _num(value: Any, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if signed:
        return f"{number:+.3f}"
    if abs(number) >= 100:
        return f"{number:,.0f}"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


_DASHBOARD_JS = """
document.querySelectorAll('[data-demo-src]').forEach((button) => {
  button.addEventListener('click', () => {
    const frame = document.getElementById('demo-frame');
    if (frame) frame.src = button.dataset.demoSrc;
  });
});
"""


_DASHBOARD_CSS = """
:root {
  color-scheme: light;
  --bg: #f6f7f4;
  --panel: #ffffff;
  --text: #18201d;
  --muted: #68736d;
  --line: #dfe4dd;
  --good: #1f7a4d;
  --warn: #a55f00;
  --bad: #b03232;
  --accent: #245a8d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.45;
}
.shell { width: min(1440px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 48px; }
.header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 18px;
}
.eyebrow { margin: 0 0 6px; color: var(--accent); font-weight: 700; text-transform: uppercase; font-size: 12px; }
h1 { margin: 0; font-size: 32px; letter-spacing: 0; }
h2 { margin: 0 0 14px; font-size: 19px; letter-spacing: 0; }
h3 { margin: 18px 0 10px; font-size: 15px; letter-spacing: 0; }
p { margin: 8px 0 0; }
.muted { color: var(--muted); }
.decision-reason { color: var(--muted); font-weight: 700; }
.status-pill {
  display: inline-flex;
  white-space: nowrap;
  border-radius: 999px;
  padding: 8px 12px;
  font-weight: 700;
  border: 1px solid var(--line);
  background: var(--panel);
}
.status-good { color: var(--good); }
.status-warn { color: var(--warn); }
.status-bad { color: var(--bad); }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 20px 0;
}
.kpi, .panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.kpi span, .small-metric span { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
.kpi b { display: block; margin-top: 8px; font-size: 25px; letter-spacing: 0; }
.kpi p { color: var(--muted); font-size: 13px; }
.kpi-good { border-top: 4px solid var(--good); }
.kpi-warn { border-top: 4px solid var(--warn); }
.kpi-bad { border-top: 4px solid var(--bad); }
.kpi-neutral { border-top: 4px solid var(--accent); }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }
.metric-row { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }
.small-metric { border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 72px; }
.small-metric b { display: block; margin-top: 6px; font-size: 20px; }
.bar-label { color: var(--muted); font-size: 13px; margin-top: 8px; }
.bar { height: 12px; border-radius: 999px; background: #e9ede7; overflow: hidden; margin: 7px 0 12px; }
.bar span { display: block; height: 100%; background: var(--accent); }
.timeline { display: grid; gap: 9px; }
.step { border-left: 5px solid var(--line); padding: 8px 10px; background: #fafbf9; border-radius: 6px; }
.step b, .step span { display: block; }
.step span, .todo span, .sources span { color: var(--muted); font-size: 13px; margin-top: 2px; }
.step-done { border-left-color: var(--good); }
.step-active { border-left-color: var(--warn); }
.step-pending { border-left-color: var(--line); }
.todo, .sources { list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }
.todo li, .sources li { padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; background: #fafbf9; }
.todo b, .todo span, .sources b, .sources span { display: block; }
.todo-done { border-left: 5px solid var(--good) !important; }
.todo-active { border-left: 5px solid var(--warn) !important; }
.todo-failed { border-left: 5px solid var(--bad) !important; }
.todo-blocked { border-left: 5px solid var(--bad) !important; opacity: 0.78; }
.todo-pending { border-left: 5px solid var(--line) !important; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 8px 7px; vertical-align: top; }
th { color: var(--muted); font-size: 11px; text-transform: uppercase; }
a { color: var(--accent); text-decoration: none; font-weight: 700; }
.proof-box { border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 12px 0; background: #fbfcfa; }
.demo-buttons { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.demo-button {
  border: 1px solid var(--line);
  background: #fdfefd;
  color: var(--text);
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 700;
}
iframe {
  width: 100%;
  min-height: 560px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #111;
}
.metric-up { color: var(--good); font-weight: 700; }
.metric-down { color: var(--bad); font-weight: 700; }
.metric-flat { color: var(--muted); font-weight: 700; }
@media (max-width: 900px) {
  .header, .two-col { display: block; }
  .status-pill { margin-top: 12px; white-space: normal; }
  .kpi-grid { grid-template-columns: 1fr 1fr; }
  .metric-row { grid-template-columns: 1fr 1fr; }
  iframe { min-height: 420px; }
}
@media (max-width: 560px) {
  .shell { width: min(100% - 20px, 1440px); padding-top: 18px; }
  .kpi-grid, .metric-row { grid-template-columns: 1fr; }
  h1 { font-size: 26px; }
  table { display: block; overflow-x: auto; white-space: nowrap; }
}
"""
