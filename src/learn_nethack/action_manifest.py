"""Action manifest for mapping NLD keypresses to active NLE action ids."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class ActionEntry:
    action_id: int
    nle_action_name: str
    raw_key_code: int
    key_label: str


@dataclass(frozen=True)
class ActionManifest:
    env_id: str
    entries: tuple[ActionEntry, ...]

    def valid_action_ids(self) -> list[int]:
        return [entry.action_id for entry in self.entries]

    def action_id_for_raw_key(self, raw_key_code: int) -> int:
        matches = [
            entry for entry in self.entries if entry.raw_key_code == raw_key_code
        ]
        if len(matches) == 1:
            return matches[0].action_id
        if len(matches) > 1:
            action_ids = [entry.action_id for entry in matches]
            raise ValueError(
                f"ambiguous raw key code {raw_key_code} maps to action ids {action_ids}"
            )
        raise KeyError(
            f"raw key code {raw_key_code} is not in active NLE action manifest"
        )

    def key_label_for_raw_key(self, raw_key_code: int) -> str:
        matches = [
            entry for entry in self.entries if entry.raw_key_code == raw_key_code
        ]
        if len(matches) == 1:
            return matches[0].key_label
        if len(matches) > 1:
            action_ids = [entry.action_id for entry in matches]
            raise ValueError(
                f"ambiguous raw key code {raw_key_code} maps to action ids {action_ids}"
            )
        raise KeyError(
            f"raw key code {raw_key_code} is not in active NLE action manifest"
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def load_action_manifest(path: str | Path) -> ActionManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ActionManifest(
        env_id=payload["env_id"],
        entries=tuple(ActionEntry(**entry) for entry in payload["entries"]),
    )


def _key_label(raw_key_code: int) -> str:
    if 32 <= raw_key_code <= 126:
        return chr(raw_key_code)
    return f"keycode_{raw_key_code}"


def _action_name(action: object) -> str:
    name = getattr(action, "name", None)
    if name is None:
        return str(action)
    return f"{action.__class__.__name__}.{name}"


def build_action_manifest_from_nle_actions(
    *, env_id: str = "NetHackChallenge-v0"
) -> ActionManifest:
    """Build the active action manifest for the requested NLE environment."""
    try:
        from nle import nethack
    except ImportError as exc:  # pragma: no cover - depends on optional NLE install.
        raise RuntimeError(
            "nle is required to build an action manifest from NLE actions"
        ) from exc

    actions = _env_actions(env_id)
    if actions is None:
        actions = getattr(nethack, "ACTIONS", None)
    if actions is None:
        raise RuntimeError("NLE actions are not available")

    entries: list[ActionEntry] = []
    for action_id, action in enumerate(actions):
        try:
            raw_key_code = int(action)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"NLE action at index {action_id} does not expose an integer key code"
            ) from exc
        entries.append(
            ActionEntry(
                action_id=action_id,
                nle_action_name=_action_name(action),
                raw_key_code=raw_key_code,
                key_label=_key_label(raw_key_code),
            )
        )

    if not entries:
        raise RuntimeError("nle.nethack.ACTIONS is empty")

    return ActionManifest(env_id=env_id, entries=tuple(entries))


def _env_actions(env_id: str) -> tuple[object, ...] | None:
    try:
        import nle  # noqa: F401
        import gymnasium as gym
    except ImportError:
        return None
    try:
        env = gym.make(env_id)
    except Exception:
        return None
    try:
        actions = getattr(getattr(env, "unwrapped", env), "actions", None)
        if actions is None:
            return None
        return tuple(actions)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
