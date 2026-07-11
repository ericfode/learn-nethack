"""Audit visible-movement pseudo labels against true NLD keypresses."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping

from learn_nethack.pseudo_labels import infer_visible_movement_pseudo_label


MOVEMENT_DIRECTIONS = frozenset({"N", "NE", "E", "SE", "S", "SW", "W", "NW"})
ProgressCallback = Callable[[dict[str, Any]], None]


def stratified_audit_gameids(
    game_metadata_by_id: Mapping[int, Mapping[str, Any]],
    *,
    max_games: int,
    seed: int,
) -> list[int]:
    """Select a deterministic role/race/alignment-stratified game cohort."""
    if max_games <= 0:
        raise ValueError("max_games must be positive")
    strata: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for gameid, metadata in sorted(game_metadata_by_id.items()):
        key = (
            str(metadata.get("role") or "unknown"),
            str(metadata.get("race") or "unknown"),
            str(metadata.get("align") or "unknown"),
        )
        strata[key].append(int(gameid))
    rng = random.Random(seed)
    for gameids in strata.values():
        rng.shuffle(gameids)

    selected: list[int] = []
    ordered_keys = sorted(strata)
    while len(selected) < max_games:
        added = False
        for key in ordered_keys:
            gameids = strata[key]
            if not gameids:
                continue
            selected.append(gameids.pop())
            added = True
            if len(selected) >= max_games:
                break
        if not added:
            break
    return sorted(selected)


def build_pseudo_label_audit_report(
    *,
    transitions: Iterable[Any],
    action_manifest: Any,
    game_metadata_by_id: Mapping[int, Mapping[str, Any]],
    max_transitions: int | None = None,
    max_examples_per_reason: int = 5,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 5_000,
) -> dict[str, Any]:
    """Compare frame-derived movement labels with mapped true keypress labels."""
    if max_transitions is not None and max_transitions <= 0:
        raise ValueError("max_transitions must be positive when provided")
    if max_examples_per_reason <= 0:
        raise ValueError("max_examples_per_reason must be positive")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")

    counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    true_actions: Counter[int] = Counter()
    pseudo_actions: Counter[int] = Counter()
    confusion: dict[int, Counter[int]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_counts: dict[str, dict[str, Counter[str]]] = {
        "role": defaultdict(Counter),
        "race": defaultdict(Counter),
        "alignment": defaultdict(Counter),
    }
    entry_by_action_id = {
        int(entry.action_id): entry for entry in action_manifest.entries
    }

    for transition in transitions:
        if (
            max_transitions is not None
            and counts["total_transitions"] >= max_transitions
        ):
            break
        counts["total_transitions"] += 1
        if (
            progress_callback is not None
            and counts["total_transitions"] % progress_interval == 0
        ):
            progress_callback(
                {
                    "schema_version": "learn-nethack.pseudo-label-audit-progress.v1",
                    "processed_transitions": counts["total_transitions"],
                    "mapped_true_keypress": counts["mapped_true_keypress"],
                    "pseudo_label_available": counts["pseudo_label_available"],
                    "comparable_transitions": counts["comparable_transitions"],
                    "exact_action_id_match": counts["exact_action_id_match"],
                    "movement_direction_equivalent": counts[
                        "movement_direction_equivalent"
                    ],
                }
            )
        metadata = game_metadata_by_id.get(int(transition.gameid), {})
        _increment_groups(group_counts, metadata, "total_transitions")

        pseudo_label = infer_visible_movement_pseudo_label(
            transition=transition,
            action_manifest=action_manifest,
        )
        if pseudo_label is None:
            rejection_reasons["pseudo_label_unavailable"] += 1
            _record_example(
                examples,
                "pseudo_label_unavailable",
                transition=transition,
                max_examples=max_examples_per_reason,
            )
        else:
            counts["pseudo_label_available"] += 1
            pseudo_actions[int(pseudo_label.action_id)] += 1
            _increment_groups(group_counts, metadata, "pseudo_label_available")
            pseudo_entry = entry_by_action_id.get(int(pseudo_label.action_id))
            if pseudo_entry is None or _movement_direction(pseudo_entry) is None:
                counts["pseudo_nonmovement_action"] += 1

        try:
            true_action_id = int(
                action_manifest.action_id_for_raw_key(int(transition.raw_key_code))
            )
        except KeyError:
            rejection_reasons["unmapped_true_keypress"] += 1
            _record_example(
                examples,
                "unmapped_true_keypress",
                transition=transition,
                pseudo_label=pseudo_label,
                max_examples=max_examples_per_reason,
            )
            continue
        except ValueError:
            rejection_reasons["ambiguous_true_keypress"] += 1
            _record_example(
                examples,
                "ambiguous_true_keypress",
                transition=transition,
                pseudo_label=pseudo_label,
                max_examples=max_examples_per_reason,
            )
            continue

        counts["mapped_true_keypress"] += 1
        true_actions[true_action_id] += 1
        _increment_groups(group_counts, metadata, "mapped_true_keypress")
        if pseudo_label is None:
            continue

        pseudo_action_id = int(pseudo_label.action_id)
        counts["comparable_transitions"] += 1
        confusion[true_action_id][pseudo_action_id] += 1
        _increment_groups(group_counts, metadata, "comparable_transitions")
        exact_match = true_action_id == pseudo_action_id
        true_entry = entry_by_action_id.get(true_action_id)
        true_direction = (
            _movement_direction(true_entry) if true_entry is not None else None
        )
        direction_equivalent = exact_match or true_direction == pseudo_label.direction

        if exact_match:
            counts["exact_action_id_match"] += 1
            counts["movement_direction_equivalent"] += 1
            _increment_groups(group_counts, metadata, "exact_action_id_match")
            _increment_groups(group_counts, metadata, "movement_direction_equivalent")
            continue
        if direction_equivalent:
            counts["movement_direction_equivalent"] += 1
            _increment_groups(group_counts, metadata, "movement_direction_equivalent")
            _record_example(
                examples,
                "direction_equivalent_action_alias",
                transition=transition,
                true_action_id=true_action_id,
                pseudo_label=pseudo_label,
                max_examples=max_examples_per_reason,
            )
            continue

        if true_direction is None:
            reason = "true_nonmovement_but_pseudo_movement"
            counts["true_nonmovement_but_pseudo_movement"] += 1
        else:
            reason = "movement_direction_mismatch"
            counts["movement_direction_mismatch"] += 1
        rejection_reasons[reason] += 1
        _record_example(
            examples,
            reason,
            transition=transition,
            true_action_id=true_action_id,
            true_direction=true_direction,
            pseudo_label=pseudo_label,
            max_examples=max_examples_per_reason,
        )

    comparable = counts["comparable_transitions"]
    mapped = counts["mapped_true_keypress"]
    exact_rate = _rate(counts["exact_action_id_match"], comparable)
    direction_rate = _rate(counts["movement_direction_equivalent"], comparable)
    pseudo_coverage = _rate(counts["pseudo_label_available"], mapped)
    dynamics_gate_passed = (
        comparable > 0
        and direction_rate >= 0.99
        and counts["pseudo_nonmovement_action"] == 0
    )
    report = {
        "schema_version": "learn-nethack.pseudo-label-audit.v1",
        "limits": {
            "max_transitions": max_transitions,
            "max_examples_per_reason": max_examples_per_reason,
        },
        "counts": dict(sorted(counts.items())),
        "rates": {
            "pseudo_label_coverage": pseudo_coverage,
            "exact_action_id_agreement": exact_rate,
            "movement_direction_equivalence": direction_rate,
        },
        "thresholds": {
            "exact_action_id_agreement": 0.95,
            "movement_direction_equivalence": 0.99,
        },
        "promotion": {
            "dynamics_conditioning_gate_passed": dynamics_gate_passed,
            "eligible_dynamics_transition_count": counts[
                "movement_direction_equivalent"
            ],
            "policy_label_interpretation": (
                "player_imitation_target"
                if comparable > 0 and exact_rate >= 0.95
                else "movement_effect_target"
            ),
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "true_action_distribution": _distribution_report(true_actions),
        "pseudo_action_distribution": _distribution_report(pseudo_actions),
        "confusion_matrix": {
            str(true_action_id): {
                str(pseudo_action_id): count
                for pseudo_action_id, count in sorted(row.items())
            }
            for true_action_id, row in sorted(confusion.items())
        },
        "breakdowns": _build_group_reports(group_counts),
        "examples": {key: value for key, value in sorted(examples.items())},
    }
    if counts["total_transitions"] <= 0:
        raise ValueError("pseudo-label audit received no transitions")
    return report


def write_pseudo_label_audit_report(
    path: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Write a deterministic local pseudo-label audit report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return target


def _movement_direction(entry: Any) -> str | None:
    suffix = str(entry.nle_action_name).rsplit(".", 1)[-1]
    return suffix if suffix in MOVEMENT_DIRECTIONS else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _distribution_report(counts: Counter[int]) -> dict[str, Any]:
    total = sum(counts.values())
    dominant_count = max(counts.values(), default=0)
    return {
        "total": total,
        "dominant_class_rate": _rate(dominant_count, total),
        "counts": {str(key): value for key, value in sorted(counts.items())},
    }


def _increment_groups(
    group_counts: dict[str, dict[str, Counter[str]]],
    metadata: Mapping[str, Any],
    key: str,
) -> None:
    for dimension, metadata_key in (
        ("role", "role"),
        ("race", "race"),
        ("alignment", "align"),
    ):
        value = str(metadata.get(metadata_key) or "unknown")
        group_counts[dimension][value][key] += 1


def _build_group_reports(
    group_counts: dict[str, dict[str, Counter[str]]],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for dimension, groups in group_counts.items():
        reports[dimension] = {}
        for value, counts in sorted(groups.items()):
            comparable = counts["comparable_transitions"]
            reports[dimension][value] = {
                "counts": dict(sorted(counts.items())),
                "exact_action_id_agreement": _rate(
                    counts["exact_action_id_match"], comparable
                ),
                "movement_direction_equivalence": _rate(
                    counts["movement_direction_equivalent"], comparable
                ),
            }
    return reports


def _record_example(
    examples: dict[str, list[dict[str, Any]]],
    reason: str,
    *,
    transition: Any,
    max_examples: int,
    true_action_id: int | None = None,
    true_direction: str | None = None,
    pseudo_label: Any | None = None,
) -> None:
    if len(examples[reason]) >= max_examples:
        return
    example = {
        "gameid": int(transition.gameid),
        "step": int(transition.step),
        "raw_key_code": int(transition.raw_key_code),
        "true_action_id": true_action_id,
        "true_direction": true_direction,
        "pseudo_action_id": (
            int(pseudo_label.action_id) if pseudo_label is not None else None
        ),
        "pseudo_direction": (
            str(pseudo_label.direction) if pseudo_label is not None else None
        ),
    }
    examples[reason].append(example)
