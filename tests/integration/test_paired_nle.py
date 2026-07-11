from __future__ import annotations

import pytest

from learn_nethack.compare_watch import seed_nle_env
from learn_nethack.paired_nle import make_paired_challenge_env


@pytest.mark.integration
def test_paired_challenge_reproduces_initial_state_with_full_action_space() -> None:
    pytest.importorskip("nle")
    first = make_paired_challenge_env(character="mon-hum-neu-mal")
    second = make_paired_challenge_env(character="mon-hum-neu-mal")
    try:
        assert len(first.actions) == 121
        assert len(second.actions) == 121
        assert seed_nle_env(first, seed=20260615)
        assert seed_nle_env(second, seed=20260615)
        first_observation, _ = first.reset()
        second_observation, _ = second.reset()
        assert (first_observation["tty_chars"] == second_observation["tty_chars"]).all()
        assert (first_observation["blstats"] == second_observation["blstats"]).all()
    finally:
        first.close()
        second.close()
