"""Pure metrics and categorical terminal-delta transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CHAR_CLASSES = 257
COLOR_VALUES = 32
COLOR_CLASSES = COLOR_VALUES + 1
CHAR_MASK_TOKEN = CHAR_CLASSES
COLOR_MASK_TOKEN = COLOR_CLASSES
TERMINAL_ROWS = 24
TERMINAL_COLS = 80


@dataclass(frozen=True)
class FrameMetricSums:
    examples: int
    cells: int
    char_correct: int
    color_correct: int
    exact_frames: int
    true_changed: int
    predicted_changed: int
    correct_changed: int
    true_positive_changed: int
    map_char_correct: int
    map_cells: int
    status_char_correct: int
    status_cells: int

    def as_metrics(self) -> dict[str, float | int]:
        precision = _ratio(self.true_positive_changed, self.predicted_changed)
        recall = _ratio(self.true_positive_changed, self.true_changed)
        changed_f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        return {
            "examples": self.examples,
            "full_frame_char_accuracy": _ratio(self.char_correct, self.cells),
            "full_frame_color_accuracy": _ratio(self.color_correct, self.cells),
            "exact_frame_rate": _ratio(self.exact_frames, self.examples),
            "changed_cell_precision": precision,
            "changed_cell_recall": recall,
            "changed_cell_f1": changed_f1,
            "changed_cell_char_accuracy": _ratio(
                self.correct_changed, self.true_changed
            ),
            "true_changed_cells": self.true_changed,
            "predicted_changed_cells": self.predicted_changed,
            "map_char_accuracy": _ratio(self.map_char_correct, self.map_cells),
            "status_char_accuracy": _ratio(self.status_char_correct, self.status_cells),
        }


def terminal_delta(current, following):
    """Encode a byte plane as unchanged=0 or replacement-byte+1."""
    import numpy as np

    current_array = np.asarray(current)
    following_array = np.asarray(following)
    if current_array.shape != following_array.shape:
        raise ValueError("current and following terminal planes must have equal shape")
    return np.where(current_array == following_array, 0, following_array + 1)


def apply_terminal_delta(current, delta):
    """Apply an unchanged/replacement categorical delta to a byte plane."""
    import numpy as np

    current_array = np.asarray(current)
    delta_array = np.asarray(delta)
    if current_array.shape != delta_array.shape:
        raise ValueError("current terminal plane and delta must have equal shape")
    return np.where(delta_array == 0, current_array, delta_array - 1).astype(
        current_array.dtype
    )


def frame_metric_sums(current, truth, prediction) -> FrameMetricSums:
    """Return additive metrics for batches of full terminal states."""
    import numpy as np

    current_chars, current_colors = current
    truth_chars, truth_colors = truth
    predicted_chars, predicted_colors = prediction
    arrays = tuple(
        np.asarray(value)
        for value in (
            current_chars,
            current_colors,
            truth_chars,
            truth_colors,
            predicted_chars,
            predicted_colors,
        )
    )
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays[1:]):
        raise ValueError("all terminal metric arrays must have equal shape")
    if len(shape) != 3 or shape[1:] != (TERMINAL_ROWS, TERMINAL_COLS):
        raise ValueError(
            f"terminal metric arrays must be [N,{TERMINAL_ROWS},{TERMINAL_COLS}]"
        )
    (
        current_chars,
        current_colors,
        truth_chars,
        truth_colors,
        predicted_chars,
        predicted_colors,
    ) = arrays
    true_changed = (truth_chars != current_chars) | (truth_colors != current_colors)
    predicted_changed = (predicted_chars != current_chars) | (
        predicted_colors != current_colors
    )
    char_equal = predicted_chars == truth_chars
    color_equal = predicted_colors == truth_colors
    exact = (char_equal & color_equal).reshape(shape[0], -1).all(axis=1)
    map_slice = slice(0, 22)
    status_slice = slice(22, 24)
    return FrameMetricSums(
        examples=int(shape[0]),
        cells=int(np.prod(shape)),
        char_correct=int(char_equal.sum()),
        color_correct=int(color_equal.sum()),
        exact_frames=int(exact.sum()),
        true_changed=int(true_changed.sum()),
        predicted_changed=int(predicted_changed.sum()),
        correct_changed=int((char_equal & true_changed).sum()),
        true_positive_changed=int((true_changed & predicted_changed).sum()),
        map_char_correct=int(char_equal[:, map_slice, :].sum()),
        map_cells=int(char_equal[:, map_slice, :].size),
        status_char_correct=int(char_equal[:, status_slice, :].sum()),
        status_cells=int(char_equal[:, status_slice, :].size),
    )


def merge_frame_metric_sums(values: Iterable[FrameMetricSums]) -> FrameMetricSums:
    items = list(values)
    if not items:
        raise ValueError("at least one frame metric sum is required")
    fields = FrameMetricSums.__dataclass_fields__
    return FrameMetricSums(
        **{name: sum(getattr(item, name) for item in items) for name in fields}
    )


def per_example_changed_f1(current, truth, prediction):
    """Return changed-cell F1 for each example in a batch."""
    import numpy as np

    current_chars, current_colors = (np.asarray(value) for value in current)
    truth_chars, truth_colors = (np.asarray(value) for value in truth)
    predicted_chars, predicted_colors = (np.asarray(value) for value in prediction)
    true_changed = (truth_chars != current_chars) | (truth_colors != current_colors)
    predicted_changed = (predicted_chars != current_chars) | (
        predicted_colors != current_colors
    )
    axes = (1, 2)
    true_positive = (true_changed & predicted_changed).sum(axis=axes)
    predicted_count = predicted_changed.sum(axis=axes)
    true_count = true_changed.sum(axis=axes)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros_like(true_positive, dtype=float),
        where=true_count != 0,
    )
    denominator = precision + recall
    return np.divide(
        2.0 * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )


def paired_bootstrap_interval(
    baseline,
    candidate,
    *,
    seed: int,
    samples: int = 5000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap a paired mean difference, candidate minus baseline."""
    import numpy as np

    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    if baseline_array.shape != candidate_array.shape or baseline_array.ndim != 1:
        raise ValueError("paired bootstrap inputs must be equal one-dimensional arrays")
    if baseline_array.size == 0:
        raise ValueError("paired bootstrap inputs must not be empty")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    differences = candidate_array - baseline_array
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    chunk_size = min(256, samples)
    offset = 0
    while offset < samples:
        count = min(chunk_size, samples - offset)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        means[offset : offset + count] = differences[indices].mean(axis=1)
        offset += count
    tail = (1.0 - confidence) / 2.0
    return {
        "pairs": int(differences.size),
        "samples": int(samples),
        "mean_difference": float(differences.mean()),
        "lower": float(np.quantile(means, tail)),
        "upper": float(np.quantile(means, 1.0 - tail)),
        "confidence": float(confidence),
    }


def contiguous_rollout_starts(
    sequence_ids,
    sequence_steps,
    candidate_indices,
    *,
    horizon: int,
    limit: int | None = None,
):
    """Return starts whose following transition rows are truly contiguous."""
    import numpy as np

    if horizon <= 0:
        raise ValueError("rollout horizon must be positive")
    sequence_ids = np.asarray(sequence_ids)
    sequence_steps = np.asarray(sequence_steps)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    allowed = set(int(index) for index in candidates.tolist())
    starts: list[int] = []
    for value in candidates:
        index = int(value)
        end = index + horizon - 1
        if end >= sequence_ids.size:
            continue
        if any(position not in allowed for position in range(index, end + 1)):
            continue
        if not (sequence_ids[index : end + 1] == sequence_ids[index]).all():
            continue
        expected = np.arange(
            int(sequence_steps[index]),
            int(sequence_steps[index]) + horizon,
        )
        if not np.array_equal(sequence_steps[index : end + 1], expected):
            continue
        starts.append(index)
        if limit is not None and len(starts) >= limit:
            break
    return np.asarray(starts, dtype=np.int64)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)
