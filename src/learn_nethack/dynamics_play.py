"""Play a learned NetHack next-frame dynamics model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from learn_nethack.action_manifest import ActionManifest, load_action_manifest
from learn_nethack.sft_rows import NEXT_FRAME_SYSTEM_PROMPT, build_next_frame_prompt


DYNAMICS_EVENT_SCHEMA_VERSION = "learn-nethack.dynamics-play-event.v1"
DYNAMICS_REPORT_SCHEMA_VERSION = "learn-nethack.dynamics-play-report.v1"
DYNAMICS_CONTRACT_SCHEMA_VERSION = "learn-nethack.dynamics-play-contract.v1"
FRAME_VALIDATION_SCHEMA_VERSION = "learn-nethack.rendered-frame-validation.v1"
REQUIRED_FRAME_SECTIONS = ("MAP", "MESSAGE", "BLSTATS", "INVENTORY")

DEFAULT_INITIAL_FRAME = "\n".join(
    [
        "MAP:",
        "@",
        "MESSAGE:",
        "Synthetic start frame for learned dynamics play.",
        "BLSTATS:",
        "<missing>",
        "INVENTORY:",
        "<missing>",
    ]
)


@dataclass(frozen=True)
class DynamicsModelSpec:
    model_name: str
    adapter_checkpoint: str | None


class NextFramePredictor(Protocol):
    def generate_next_frame_json(
        self,
        *,
        observation_text: str,
        action_id: int,
        history: list[tuple[str, int]],
    ) -> str:
        """Return the model's raw completion for one next-frame step."""


def build_dynamics_messages(
    *,
    observation_text: str,
    action_id: int,
    history: list[tuple[str, int]],
) -> list[dict[str, str]]:
    """Build the chat messages used to query the next-frame dynamics task."""
    return [
        {"role": "system", "content": NEXT_FRAME_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_next_frame_prompt(
                observation_text=observation_text,
                action_id=action_id,
                history=history,
            ),
        },
    ]


def parse_next_frame_response(response_text: str) -> str:
    """Parse a next-frame model response.

    New dynamics rows train the assistant to emit raw rendered frame text. Legacy
    rows and checkpoints may still emit exact ``{"next_frame": str}`` JSON.
    """
    stripped = response_text.strip()
    if _looks_like_raw_frame(stripped):
        return stripped
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid next-frame JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"next_frame"}:
        raise ValueError("expected only next_frame in model output")
    next_frame = payload["next_frame"]
    if not isinstance(next_frame, str):
        raise ValueError("next_frame must be a string")
    return next_frame


def _looks_like_raw_frame(text: str) -> bool:
    return text.startswith("MAP:")


def read_initial_frame(
    *,
    initial_frame_path: str | Path | None,
    initial_row_path: str | Path | None,
) -> str:
    """Read the starting rendered frame for dynamics play."""
    if initial_frame_path is not None and initial_row_path is not None:
        raise ValueError("use only one of initial_frame_path or initial_row_path")
    if initial_frame_path is not None:
        return Path(initial_frame_path).read_text(encoding="utf-8").rstrip("\n")
    if initial_row_path is not None:
        return _read_initial_frame_from_row(initial_row_path)
    return DEFAULT_INITIAL_FRAME


def read_ground_truth_frames_from_next_frame_rows(
    path: str | Path,
    *,
    max_frames: int | None = None,
) -> list[str]:
    """Read assistant next-frame labels from a next_frame JSONL dataset."""
    frames: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if max_frames is not None and len(frames) >= max_frames:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "next_frame":
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError("next_frame row does not contain assistant label")
            assistant_content = messages[2].get("content")
            if not isinstance(assistant_content, str):
                raise ValueError("next_frame row assistant label is not text")
            frames.append(parse_next_frame_response(assistant_content))
    if not frames:
        raise ValueError(f"no next_frame ground-truth labels found in {path}")
    return frames


def parse_action_id_list(action_ids: str) -> list[int]:
    """Parse comma-separated action IDs for scripted dynamics play."""
    values: list[int] = []
    for part in action_ids.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            values.append(int(stripped))
        except ValueError as exc:
            raise ValueError(f"invalid action id {stripped!r}") from exc
    if not values:
        raise ValueError("at least one action id is required")
    return values


class TransformerNextFramePredictor:
    """Generate next-frame text with a base Gemma model and optional adapter."""

    def __init__(
        self,
        spec: DynamicsModelSpec,
        *,
        device: str | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        torch_dtype: str = "auto",
    ) -> None:
        self.spec = spec
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.torch_dtype = torch_dtype
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._torch: Any | None = None

    def generate_next_frame_json(
        self,
        *,
        observation_text: str,
        action_id: int,
        history: list[tuple[str, int]],
    ) -> str:
        model, tokenizer, torch = self._load()
        messages = build_dynamics_messages(
            observation_text=observation_text,
            action_id=action_id,
            history=history,
        )
        prompt = _apply_chat_template(tokenizer, messages)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = _model_device(model)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            generation_kwargs["temperature"] = self.temperature
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        elif eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id
        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)
        input_length = int(encoded["input_ids"].shape[1])
        completion_ids = output_ids[:, input_length:]
        return tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

    def score_next_frame_response(
        self,
        *,
        observation_text: str,
        action_id: int,
        target_response: str,
        history: list[tuple[str, int]],
    ) -> dict[str, float]:
        """Score the exact next-frame target without free-form generation."""
        model, tokenizer, torch = self._load()
        messages = build_dynamics_messages(
            observation_text=observation_text,
            action_id=action_id,
            history=history,
        )
        prompt = _apply_chat_template(tokenizer, messages)
        prompt_ids = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        target_ids = tokenizer(
            target_response,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        if int(target_ids.shape[1]) == 0:
            raise ValueError("target_response must tokenize to at least one token")
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        labels = input_ids.clone()
        labels[:, : int(prompt_ids.shape[1])] = -100
        attention_mask = torch.ones_like(input_ids)
        device = _model_device(model)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = labels[:, 1:].to(device)
        target_mask = shifted_labels != -100
        safe_labels = shifted_labels.masked_fill(~target_mask, 0)
        token_log_probs = torch.log_softmax(shifted_logits, dim=-1)
        selected_log_probs = token_log_probs.gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        negative_log_likelihood = -selected_log_probs[target_mask].sum()
        argmax_matches = (
            (shifted_logits.argmax(dim=-1) == shifted_labels) & target_mask
        ).sum()
        return {
            "token_count": float(target_mask.sum().item()),
            "negative_log_likelihood": float(negative_log_likelihood.item()),
            "argmax_match_count": float(argmax_matches.item()),
        }

    def _load(self) -> tuple[Any, Any, Any]:
        if (
            self._model is not None
            and self._tokenizer is not None
            and self._torch is not None
        ):
            return self._model, self._tokenizer, self._torch

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - requires model deps.
            raise RuntimeError(
                "torch and transformers are required for dynamics play; "
                "run with `uv run --extra modal-train ...` or inside the Modal image"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)
        model_kwargs: dict[str, Any] = {}
        if self.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        if self.device is None:
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_name,
            **model_kwargs,
        )
        if self.spec.adapter_checkpoint is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - requires model deps.
                raise RuntimeError(
                    "peft is required to load the next-frame adapter checkpoint"
                ) from exc
            model = PeftModel.from_pretrained(model, self.spec.adapter_checkpoint)
        if self.device is not None:
            model = model.to(self.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        return model, tokenizer, torch


def run_scripted_dynamics_session(
    *,
    run_id: str,
    model_spec: DynamicsModelSpec,
    predictor: NextFramePredictor,
    initial_frame: str,
    action_ids: Sequence[int],
    ground_truth_frames: Sequence[str | None] | None = None,
    action_manifest: ActionManifest,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Run a deterministic action script through the learned dynamics model."""
    target = _prepare_output_dir(out_dir)
    writer = _SessionWriter(
        run_id=run_id,
        model_spec=model_spec,
        action_manifest=action_manifest,
        out_dir=target,
    )
    current_frame = initial_frame
    history: list[tuple[str, int]] = []
    status = "completed"
    with writer:
        for step_index, action_id in enumerate(action_ids):
            event = _predict_event(
                run_id=run_id,
                step_index=step_index,
                predictor=predictor,
                current_frame=current_frame,
                action_id=action_id,
                history=history,
                ground_truth_frame=_ground_truth_frame_at(
                    ground_truth_frames,
                    step_index,
                ),
                action_manifest=action_manifest,
            )
            writer.write_event(event)
            if event["status"] != "predicted":
                status = "stopped_on_parse_failure"
                break
            history.append((current_frame, int(action_id)))
            current_frame = str(event["predicted_frame"])
    return writer.write_report(status=status)


def run_interactive_dynamics_session(
    *,
    run_id: str,
    model_spec: DynamicsModelSpec,
    predictor: NextFramePredictor,
    initial_frame: str,
    action_manifest: ActionManifest,
    out_dir: str | Path,
    max_steps: int,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run an interactive terminal loop against the learned dynamics model."""
    target = _prepare_output_dir(out_dir)
    writer = _SessionWriter(
        run_id=run_id,
        model_spec=model_spec,
        action_manifest=action_manifest,
        out_dir=target,
    )
    current_frame = initial_frame
    history: list[tuple[str, int]] = []
    status = "completed"
    output_fn(current_frame)
    output_fn("Enter an action_id, or :quit.")
    with writer:
        for step_index in range(max_steps):
            command = input_fn("action_id> ").strip()
            if command in {":q", ":quit", "quit", "exit"}:
                status = "user_stopped"
                break
            try:
                action_id = int(command)
            except ValueError:
                output_fn("invalid action_id")
                continue
            try:
                event = _predict_event(
                    run_id=run_id,
                    step_index=step_index,
                    predictor=predictor,
                    current_frame=current_frame,
                    action_id=action_id,
                    history=history,
                    ground_truth_frame=None,
                    action_manifest=action_manifest,
                )
            except ValueError as exc:
                output_fn(str(exc))
                continue
            writer.write_event(event)
            if event["status"] != "predicted":
                output_fn(str(event["error"]))
                status = "stopped_on_parse_failure"
                break
            history.append((current_frame, action_id))
            current_frame = str(event["predicted_frame"])
            output_fn(current_frame)
        else:
            status = "max_steps_reached"
    return writer.write_report(status=status)


def write_dynamics_play_contract(
    *,
    run_id: str,
    adapter_checkpoint: str | None,
    action_manifest_path: str | Path,
    out_dir: str | Path,
    model_name: str = "google/gemma-4-E4b-it",
    initial_frame_path: str | Path | None = None,
    initial_row_path: str | Path | None = None,
    ground_truth_rows_path: str | Path | None = None,
    max_steps: int = 80,
) -> dict[str, Any]:
    """Write a dry-run contract without importing model dependencies."""
    manifest = load_action_manifest(action_manifest_path)
    target = _prepare_output_dir(out_dir)
    contract = {
        "schema_version": DYNAMICS_CONTRACT_SCHEMA_VERSION,
        "run_id": run_id,
        "model": asdict(
            DynamicsModelSpec(
                model_name=model_name,
                adapter_checkpoint=adapter_checkpoint,
            )
        ),
        "action_manifest_path": str(action_manifest_path),
        "valid_action_ids": manifest.valid_action_ids(),
        "initial_frame_path": (
            None if initial_frame_path is None else str(initial_frame_path)
        ),
        "initial_row_path": None if initial_row_path is None else str(initial_row_path),
        "ground_truth_rows_path": (
            None if ground_truth_rows_path is None else str(ground_truth_rows_path)
        ),
        "max_steps": max_steps,
        "status": "requires_model_deps_and_checkpoint_to_run",
    }
    (target / "dynamics_play_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def _read_initial_frame_from_row(path: str | Path) -> str:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") != "next_frame":
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError("next_frame row does not contain user prompt")
            user_content = messages[1].get("content")
            if not isinstance(user_content, str):
                raise ValueError("next_frame row user prompt is not text")
            return _extract_current_observation(user_content)
    raise ValueError(f"no next_frame rows found in {path}")


def _extract_current_observation(user_prompt: str) -> str:
    marker = "Current observation:\n"
    if marker not in user_prompt:
        raise ValueError("next_frame row does not contain Current observation")
    return user_prompt.split(marker, 1)[1].rstrip("\n")


def validate_rendered_nethack_frame(
    frame: str,
    *,
    ground_truth_frame: str | None = None,
) -> dict[str, Any]:
    """Validate a rendered frame and optionally compare it with NLE ground truth."""
    sections = _frame_sections(frame)
    missing_sections = [
        section for section in REQUIRED_FRAME_SECTIONS if section not in sections
    ]
    map_text = _section_text(sections, "MAP")
    map_lines = [line for line in sections.get("MAP", ()) if line.strip()]
    blstats = _parse_blstats(sections)
    blstats_values = blstats["values"]
    blstats_is_numeric = bool(
        isinstance(blstats_values, list)
        and all(isinstance(value, int | float) for value in blstats_values)
    )
    blstats_length = len(blstats_values) if isinstance(blstats_values, list) else None
    rendered_frame_parse_valid = not missing_sections
    nle_observation_shape_valid = bool(
        rendered_frame_parse_valid
        and map_lines
        and "@" in map_text
        and blstats_is_numeric
        and blstats_length == 27
    )
    validation: dict[str, Any] = {
        "schema_version": FRAME_VALIDATION_SCHEMA_VERSION,
        "rendered_frame_parse_valid": rendered_frame_parse_valid,
        "nle_observation_shape_valid": nle_observation_shape_valid,
        "missing_sections": missing_sections,
        "map_nonempty_line_count": len(map_lines),
        "map_has_player_glyph": "@" in map_text,
        "blstats_parse_valid": blstats["parse_valid"],
        "blstats_length": blstats_length,
        "blstats_length_valid": blstats_length == 27,
        "ground_truth_available": ground_truth_frame is not None,
        "ground_truth_exact_match": None,
        "char_accuracy": None,
        "map_exact_match": None,
        "message_exact_match": None,
        "blstats_exact_match": None,
    }
    if ground_truth_frame is None:
        return validation

    ground_truth_sections = _frame_sections(ground_truth_frame)
    validation.update(
        {
            "ground_truth_exact_match": frame == ground_truth_frame,
            "char_accuracy": _char_accuracy(frame, ground_truth_frame),
            "map_exact_match": _section_text(sections, "MAP")
            == _section_text(ground_truth_sections, "MAP"),
            "message_exact_match": _section_text(sections, "MESSAGE")
            == _section_text(ground_truth_sections, "MESSAGE"),
            "blstats_exact_match": _section_text(sections, "BLSTATS")
            == _section_text(ground_truth_sections, "BLSTATS"),
        }
    )
    return validation


def _ground_truth_frame_at(
    ground_truth_frames: Sequence[str | None] | None,
    step_index: int,
) -> str | None:
    if ground_truth_frames is None or step_index >= len(ground_truth_frames):
        return None
    return ground_truth_frames[step_index]


def _frame_sections(frame: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in frame.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in REQUIRED_FRAME_SECTIONS:
            current_section = stripped[:-1]
            sections.setdefault(current_section, [])
            continue
        if current_section is not None:
            sections[current_section].append(line)
    return sections


def _section_text(sections: dict[str, list[str]], section: str) -> str:
    return "\n".join(sections.get(section, [])).strip("\n")


def _parse_blstats(sections: dict[str, list[str]]) -> dict[str, Any]:
    lines = [line.strip() for line in sections.get("BLSTATS", ()) if line.strip()]
    if not lines:
        return {"parse_valid": False, "values": None}
    try:
        values = json.loads(lines[0])
    except json.JSONDecodeError:
        return {"parse_valid": False, "values": None}
    return {"parse_valid": isinstance(values, list), "values": values}


def _char_accuracy(prediction: str, ground_truth: str) -> float:
    total = 0
    matching = 0
    for pred_char, truth_char in zip_longest(
        prediction,
        ground_truth,
        fillvalue=None,
    ):
        total += 1
        if pred_char == truth_char:
            matching += 1
    if total == 0:
        return 1.0
    return matching / total


def _predict_event(
    *,
    run_id: str,
    step_index: int,
    predictor: NextFramePredictor,
    current_frame: str,
    action_id: int,
    history: list[tuple[str, int]],
    ground_truth_frame: str | None,
    action_manifest: ActionManifest,
) -> dict[str, Any]:
    valid_action_ids = set(action_manifest.valid_action_ids())
    if int(action_id) not in valid_action_ids:
        raise ValueError(f"action_id {action_id} is not in active action manifest")
    raw_output = predictor.generate_next_frame_json(
        observation_text=current_frame,
        action_id=int(action_id),
        history=list(history),
    )
    event = {
        "schema_version": DYNAMICS_EVENT_SCHEMA_VERSION,
        "run_id": run_id,
        "step": step_index,
        "action_id": int(action_id),
        "action_label": _action_label(action_manifest, int(action_id)),
        "history_length": len(history),
        "prompt_frame": current_frame,
        "ground_truth_frame": ground_truth_frame,
        "raw_model_output": raw_output,
    }
    try:
        next_frame = parse_next_frame_response(raw_output)
    except ValueError as exc:
        return {
            **event,
            "status": "parse_failed",
            "error": str(exc),
            "predicted_frame": None,
            "validation": validate_rendered_nethack_frame(
                "",
                ground_truth_frame=ground_truth_frame,
            ),
        }
    return {
        **event,
        "status": "predicted",
        "error": None,
        "predicted_frame": next_frame,
        "validation": validate_rendered_nethack_frame(
            next_frame,
            ground_truth_frame=ground_truth_frame,
        ),
    }


def _action_label(action_manifest: ActionManifest, action_id: int) -> str | None:
    for entry in action_manifest.entries:
        if entry.action_id == action_id:
            return entry.key_label
    return None


def _prepare_output_dir(out_dir: str | Path) -> Path:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


class _SessionWriter:
    def __init__(
        self,
        *,
        run_id: str,
        model_spec: DynamicsModelSpec,
        action_manifest: ActionManifest,
        out_dir: Path,
    ) -> None:
        self.run_id = run_id
        self.model_spec = model_spec
        self.action_manifest = action_manifest
        self.out_dir = out_dir
        self.events_path = out_dir / "events.jsonl"
        self.latest_path = out_dir / "latest.json"
        self.report_path = out_dir / "report.json"
        self.viewer_path = out_dir / "index.html"
        self.events: list[dict[str, Any]] = []
        self._handle: Any | None = None

    def __enter__(self) -> "_SessionWriter":
        self._handle = self.events_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        if self._handle is not None:
            self._handle.close()

    def write_event(self, event: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("session writer is not open")
        line = json.dumps(event, sort_keys=True)
        self._handle.write(line + "\n")
        self._handle.flush()
        self.latest_path.write_text(line + "\n", encoding="utf-8")
        self.events.append(event)

    def write_report(self, *, status: str) -> dict[str, Any]:
        report = {
            "schema_version": DYNAMICS_REPORT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "model": asdict(self.model_spec),
            "action_manifest": {
                "env_id": self.action_manifest.env_id,
                "valid_action_ids": self.action_manifest.valid_action_ids(),
            },
            "event_count": len(self.events),
            "events_path": str(self.events_path),
            "latest_path": str(self.latest_path),
            "viewer_path": str(self.viewer_path),
            "status": status,
        }
        self.report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.viewer_path.write_text(
            _render_static_viewer(report=report, events=self.events),
            encoding="utf-8",
        )
        return report


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        + "\nassistant: "
    )


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _render_static_viewer(
    *, report: dict[str, Any], events: Sequence[dict[str, Any]]
) -> str:
    report_json = json.dumps(report, sort_keys=True)
    events_json = json.dumps(list(events), sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NetHack Dynamics Play</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101214;
      --panel: #181d21;
      --line: #34404a;
      --text: #e7ecef;
      --muted: #a7b0b7;
      --accent: #67c1a3;
      --bad: #d66f6f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 16px 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px 24px;
      align-items: baseline;
    }}
    h1 {{
      font-size: 18px;
      margin: 0;
    }}
    .meta {{
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
    }}
    main {{
      padding: 16px 20px 24px;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(160px, 1fr) auto;
      gap: 16px;
      align-items: center;
      margin-bottom: 16px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}
    section {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px;
      min-width: 0;
      overflow: hidden;
    }}
    section h2 {{
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      color: var(--accent);
      font-size: 14px;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 1px;
      margin: 0 0 16px;
      background: var(--line);
    }}
    dt, dd {{
      margin: 0;
      background: var(--panel);
      padding: 8px 10px;
      min-width: 0;
    }}
    dt {{
      color: var(--muted);
    }}
    dd.parse_failed {{
      color: var(--bad);
    }}
    pre {{
      margin: 0;
      padding: 12px;
      min-height: 340px;
      max-height: 70vh;
      overflow: auto;
      white-space: pre-wrap;
      color: #f4f7f9;
      background: #080a0b;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      dl {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>NetHack Dynamics Play</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <div class="controls">
      <input id="step" type="range" min="0" value="0">
      <strong id="stepLabel"></strong>
    </div>
    <dl id="stats"></dl>
    <div class="grid">
      <section>
        <h2>Prompt Frame</h2>
        <pre id="promptFrame"></pre>
      </section>
      <section>
        <h2>Predicted Next Frame</h2>
        <pre id="predictedFrame"></pre>
      </section>
      <section>
        <h2>Ground Truth Next Frame</h2>
        <pre id="groundTruthFrame"></pre>
      </section>
    </div>
  </main>
  <script type="application/json" id="report-data">{report_json}</script>
  <script type="application/json" id="event-data">{events_json}</script>
  <script>
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const events = JSON.parse(document.getElementById("event-data").textContent);
    const slider = document.getElementById("step");
    const stepLabel = document.getElementById("stepLabel");
    slider.max = Math.max(events.length - 1, 0);
    document.getElementById("meta").textContent =
      `run ${{report.run_id}} | events ${{events.length}} | ${{report.status}}`;

    function statsHtml(event) {{
      const validation = event.validation || {{}};
      const charAccuracy = typeof validation.char_accuracy === "number"
        ? validation.char_accuracy.toFixed(3)
        : "";
      const pairs = [
        ["action", event.action_id ?? ""],
        ["label", event.action_label ?? ""],
        ["status", event.status ?? ""],
        ["history", event.history_length ?? 0],
        ["shape valid", validation.nle_observation_shape_valid ?? ""],
        ["truth exact", validation.ground_truth_exact_match ?? ""],
        ["char acc", charAccuracy],
        ["error", event.error ?? ""],
      ];
      return pairs.map(([key, value]) => {{
        const className = key === "status" && value !== "predicted" ? "parse_failed" : "";
        return `<dt>${{key}}</dt><dd class="${{className}}">${{value}}</dd>`;
      }}).join("");
    }}

    function render(index) {{
      const event = events[index] || {{}};
      stepLabel.textContent = `step ${{event.step ?? 0}}`;
      document.getElementById("stats").innerHTML = statsHtml(event);
      document.getElementById("promptFrame").textContent = event.prompt_frame || "";
      document.getElementById("predictedFrame").textContent = event.predicted_frame || "";
      document.getElementById("groundTruthFrame").textContent = event.ground_truth_frame || "";
    }}
    slider.addEventListener("input", () => render(Number(slider.value)));
    render(0);
  </script>
</body>
</html>
"""
