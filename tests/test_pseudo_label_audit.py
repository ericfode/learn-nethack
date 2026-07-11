from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from learn_nethack.action_manifest import ActionEntry, ActionManifest
from learn_nethack.nld_decode import DecodedTransition
from learn_nethack.pseudo_label_audit import (
    build_pseudo_label_audit_report,
    stratified_audit_gameids,
    write_pseudo_label_audit_report,
)


def _observation(row: int, col: int) -> dict:
    chars = [[32 for _ in range(5)] for _ in range(3)]
    chars[row][col] = 64
    return {"tty_chars": chars}


def _transition(raw_key: int, *, start=(1, 1), end=(1, 2), step=0):
    return DecodedTransition(
        gameid=1,
        step=step,
        raw_key_code=raw_key,
        observation=_observation(*start),
        next_observation=_observation(*end),
    )


def _manifest() -> ActionManifest:
    return ActionManifest(
        env_id="NetHack-v0",
        entries=(
            ActionEntry(0, "CompassDirection.N", 107, "k"),
            ActionEntry(1, "CompassDirection.E", 108, "l"),
            ActionEntry(2, "MiscDirection.WAIT", 46, "."),
            ActionEntry(3, "CompassDirection.E", 76, "L"),
        ),
    )


def test_pseudo_label_audit_separates_exact_equivalent_and_wrong_labels() -> None:
    transitions = [
        _transition(108, step=1),
        _transition(76, step=2),
        _transition(107, step=3),
        _transition(46, step=4),
        _transition(108, start=(1, 1), end=(1, 1), step=5),
        _transition(999, step=6),
    ]

    report = build_pseudo_label_audit_report(
        transitions=transitions,
        action_manifest=_manifest(),
        game_metadata_by_id={1: {"role": "Mon", "race": "Hum", "align": "Neu"}},
    )

    assert report["counts"]["total_transitions"] == 6
    assert report["counts"]["mapped_true_keypress"] == 5
    assert report["counts"]["comparable_transitions"] == 4
    assert report["counts"]["exact_action_id_match"] == 1
    assert report["counts"]["movement_direction_equivalent"] == 2
    assert report["rates"]["exact_action_id_agreement"] == 0.25
    assert report["rates"]["movement_direction_equivalence"] == 0.5
    assert (
        report["promotion"]["policy_label_interpretation"] == "movement_effect_target"
    )
    assert not report["promotion"]["dynamics_conditioning_gate_passed"]
    assert report["rejection_reasons"] == {
        "movement_direction_mismatch": 1,
        "pseudo_label_unavailable": 1,
        "true_nonmovement_but_pseudo_movement": 1,
        "unmapped_true_keypress": 1,
    }
    assert report["confusion_matrix"]["3"]["1"] == 1
    assert report["breakdowns"]["role"]["Mon"]["movement_direction_equivalence"] == 0.5
    assert "true_nonmovement_but_pseudo_movement" in report["examples"]


def test_pseudo_label_audit_passes_matched_movement_gate_and_writes_json() -> None:
    report = build_pseudo_label_audit_report(
        transitions=[_transition(108, step=1), _transition(108, step=2)],
        action_manifest=_manifest(),
        game_metadata_by_id={},
    )

    assert report["promotion"]["dynamics_conditioning_gate_passed"]
    assert (
        report["promotion"]["policy_label_interpretation"] == "player_imitation_target"
    )
    assert report["pseudo_action_distribution"]["dominant_class_rate"] == 1.0

    with TemporaryDirectory() as tmp:
        path = write_pseudo_label_audit_report(Path(tmp) / "audit.json", report)
        written = json.loads(path.read_text(encoding="utf-8"))

    assert written["schema_version"] == "learn-nethack.pseudo-label-audit.v1"


def test_pseudo_label_audit_rejects_empty_transition_stream() -> None:
    try:
        build_pseudo_label_audit_report(
            transitions=[],
            action_manifest=_manifest(),
            game_metadata_by_id={},
        )
    except ValueError as exc:
        assert "no transitions" in str(exc)
    else:
        raise AssertionError("empty audit must fail")


def test_stratified_audit_gameids_is_deterministic_and_covers_roles() -> None:
    metadata = {
        1: {"role": "Arc", "race": "Hum", "align": "Law"},
        2: {"role": "Arc", "race": "Hum", "align": "Law"},
        3: {"role": "Mon", "race": "Hum", "align": "Neu"},
        4: {"role": "Mon", "race": "Hum", "align": "Neu"},
        5: {"role": "Wiz", "race": "Elf", "align": "Cha"},
        6: {"role": "Wiz", "race": "Elf", "align": "Cha"},
    }

    first = stratified_audit_gameids(metadata, max_games=3, seed=17)
    second = stratified_audit_gameids(metadata, max_games=3, seed=17)

    assert first == second
    assert {metadata[gameid]["role"] for gameid in first} == {"Arc", "Mon", "Wiz"}
