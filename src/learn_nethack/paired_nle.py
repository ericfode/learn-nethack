"""Seedable NLE Challenge-equivalent environment for paired model evaluation."""

from __future__ import annotations

from typing import Any


PAIRED_CHALLENGE_ENV_ID = "NetHackPairedChallenge-v0"
CHALLENGE_MANIFEST_ENV_ID = "NetHackChallenge-v0"


def action_manifest_env_id_for(env_id: str) -> str:
    if env_id == PAIRED_CHALLENGE_ENV_ID:
        return CHALLENGE_MANIFEST_ENV_ID
    return env_id


def make_paired_challenge_env(*, character: str) -> Any:
    """Build Challenge behavior with seedable NLE initialization."""
    try:
        from nle import nethack
        from nle.env.tasks import NetHackChallenge, NetHackScore
    except ImportError as exc:  # pragma: no cover - optional local dependency.
        raise RuntimeError(
            "nle is required to construct NetHackPairedChallenge-v0"
        ) from exc

    class SeedableNetHackChallenge(NetHackChallenge):
        def __init__(
            self,
            *args,
            character: str = "@",
            allow_all_yn_questions: bool = True,
            allow_all_modes: bool = True,
            penalty_mode: str = "constant",
            penalty_step: float = -0.0,
            penalty_time: float = -0.0,
            max_episode_steps: int = 1_000_000,
            observation_keys: tuple[str, ...] = (
                "glyphs",
                "chars",
                "colors",
                "specials",
                "blstats",
                "message",
                "inv_glyphs",
                "inv_strs",
                "inv_letters",
                "inv_oclasses",
                "tty_chars",
                "tty_colors",
                "tty_cursor",
                "misc",
            ),
            no_progress_timeout: int = 10_000,
            **kwargs,
        ) -> None:
            kwargs["wizard"] = False
            NetHackScore.__init__(
                self,
                *args,
                actions=nethack.ACTIONS,
                character=character,
                allow_all_yn_questions=allow_all_yn_questions,
                allow_all_modes=allow_all_modes,
                penalty_mode=penalty_mode,
                penalty_step=penalty_step,
                penalty_time=penalty_time,
                max_episode_steps=max_episode_steps,
                observation_keys=observation_keys,
                **kwargs,
            )
            self.no_progress_timeout = no_progress_timeout

        def seed(
            self,
            core: int | None = None,
            disp: int | None = None,
            reseed: bool = True,
            lgen: int | None = None,
        ):
            return NetHackScore.seed(
                self,
                core=core,
                disp=disp,
                reseed=reseed,
                lgen=lgen,
            )

    return SeedableNetHackChallenge(character=character)
