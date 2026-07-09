from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from learn_nethack.action_manifest import build_action_manifest_from_nle_actions
from learn_nethack.nld_decode import iter_nld_ttyrec_batches
from learn_nethack.nld_metadata import inspect_nld_db
from learn_nethack.nld_metadata import read_game_metadata, read_gameids
from learn_nethack.sft_build import build_sft_from_decoded_batches


TASTER_DB = Path("/Users/ericfode/data/nld/nld-aa-taster/ttyrecs.db")


@pytest.mark.integration
def test_nld_taster_metadata_available() -> None:
    if not TASTER_DB.exists():
        pytest.skip(f"{TASTER_DB} is not available")

    report = inspect_nld_db(TASTER_DB)

    assert report.dataset_name == "nld-aa-taster"
    assert report.ttyrec_version == 3
    assert report.game_count > 0
    assert report.ttyrec_count > 0


@pytest.mark.integration
def test_nle_dataset_dependency_gate_is_explicit() -> None:
    if (
        importlib.util.find_spec("nle") is None
        or importlib.util.find_spec("nle.dataset") is None
    ):
        pytest.skip("nle.dataset is required for NLD ttyrec decoding")

    import nle.dataset  # noqa: F401


@pytest.mark.integration
def test_nld_taster_builds_multitask_rows_from_real_ttyrec(tmp_path: Path) -> None:
    if not TASTER_DB.exists():
        pytest.skip(f"{TASTER_DB} is not available")
    if (
        importlib.util.find_spec("nle") is None
        or importlib.util.find_spec("nle.dataset") is None
    ):
        pytest.skip("nle.dataset is required for NLD ttyrec decoding")

    report = inspect_nld_db(TASTER_DB)
    gameids = read_gameids(TASTER_DB)[:16]
    batches = iter_nld_ttyrec_batches(
        dataset_name=report.dataset_name,
        batch_size=4,
        seq_length=128,
        dbfilename=str(TASTER_DB),
        gameids=gameids,
        shuffle=False,
        loop_forever=False,
    )

    result = build_sft_from_decoded_batches(
        dataset_name=report.dataset_name,
        mode="single_frame",
        batches=batches,
        action_manifest=build_action_manifest_from_nle_actions(),
        gameids=gameids,
        game_metadata_by_id=read_game_metadata(TASTER_DB),
        out_dir=tmp_path,
        max_rows=64,
        seed=20260615,
        tasks=("policy_action", "next_frame"),
    )

    assert result.accepted_policy_rows == 64
    assert result.accepted_next_frame_rows > 0
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
