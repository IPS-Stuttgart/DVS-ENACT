"""MEVDT loading and event-support diagnostics."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

MEVDT_DATASET_URL = "https://deepblue.lib.umich.edu/data/concern/data_sets/bc386k045"
MEVDT_DOI = "https://doi.org/10.7302/d5k3-9150"


@dataclass(frozen=True)
class EventBatch:
    """DVS events represented as timestamp, pixel coordinate, and polarity arrays."""

    ts: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray

    @property
    def count(self) -> int:
        return int(self.ts.shape[0])


@dataclass(frozen=True)
class BoundingBox:
    """One object bounding-box annotation."""

    frame: int
    track_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    timestamp_ns: int | None = None
    class_label: str | None = None

    @property
    def width(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return float(self.y_max - self.y_min)

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max)],
            dtype=float,
        )

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)


@dataclass(frozen=True)
class TrackWindowDiagnostics:
    """Event-support diagnostics for one labeled track interval."""

    track_id: int
    frame: int
    next_frame: int
    timestamp_ns: int | None
    next_timestamp_ns: int | None
    bbox: dict[str, float]
    center_velocity_px_per_frame: list[float]
    event_count: int
    side_band_counts: dict[str, int]
    active_side_fraction: float
    inactive_side_fraction: float
    event_bbox_width_ratio: float | None
    event_bbox_height_ratio: float | None
    event_bbox_area_ratio: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def find_event_csv_files(dataset_root: str | Path) -> list[Path]:
    """Return sequence-long event CSV files below a MEVDT extraction root."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"MEVDT root does not exist: {root}")
    candidates = sorted(root.glob("sequences/**/*.csv"))
    if candidates:
        return candidates
    return sorted(
        path
        for path in root.rglob("*.csv")
        if "label" not in {part.lower() for part in path.parts}
    )


def find_tracking_label_files(dataset_root: str | Path) -> list[Path]:
    """Return likely tracking-label files below a MEVDT extraction root."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"MEVDT root does not exist: {root}")
    preferred_roots = [
        path
        for path in root.rglob("*")
        if path.is_dir() and "tracking" in path.name.lower()
    ]
    search_roots = preferred_roots if preferred_roots else [root]
    files: list[Path] = []
    for search_root in search_roots:
        files.extend(
            path
            for path in search_root.rglob("*")
            if path.suffix.lower() in {".txt", ".csv", ".json"}
        )
    return sorted(files)


def _parse_float_token(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _split_row(line: str) -> list[str]:
    return [token for token in re.split(r"[\s,]+", line.strip()) if token]


def read_event_csv(
    path: str | Path,
    start_ns: int | None = None,
    end_ns: int | None = None,
    bbox: BoundingBox | None = None,
    max_events: int | None = None,
) -> EventBatch:
    """Read MEVDT event CSV rows in documented ``ts,x,y,p`` format."""
    timestamps: list[int] = []
    xs: list[int] = []
    ys: list[int] = []
    polarities: list[int] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or len(row) < 4:
                continue
            parsed = [_parse_float_token(token.strip()) for token in row[:4]]
            if any(value is None for value in parsed):
                continue
            ts, x_value, y_value, polarity = parsed
            ts_int = int(ts)
            x_int = int(x_value)
            y_int = int(y_value)
            if start_ns is not None and ts_int < start_ns:
                continue
            if end_ns is not None and ts_int >= end_ns:
                continue
            if bbox is not None and not (
                bbox.x_min <= x_int <= bbox.x_max and bbox.y_min <= y_int <= bbox.y_max
            ):
                continue
            timestamps.append(ts_int)
            xs.append(x_int)
            ys.append(y_int)
            polarities.append(int(polarity))
            if max_events is not None and len(timestamps) >= max_events:
                break
    return EventBatch(
        ts=np.array(timestamps, dtype=np.int64),
        x=np.array(xs, dtype=np.int32),
        y=np.array(ys, dtype=np.int32),
        p=np.array(polarities, dtype=np.int8),
    )


def _bbox_from_header_row(header: list[str], row: list[str], row_index: int) -> BoundingBox:
    values = {name.lower(): value for name, value in zip(header, row, strict=False)}

    def first_float(*names: str, default: float | None = None) -> float:
        for name in names:
            if name.lower() in values:
                parsed = _parse_float_token(values[name.lower()])
                if parsed is not None:
                    return parsed
        if default is None:
            raise ValueError(f"Could not find any of columns {names}")
        return default

    frame = int(first_float("frame", "frame_id", "image_id", default=row_index))
    track_id = int(first_float("track_id", "id", "obj_id", "object_id", default=-1))
    x_min = first_float("x_min", "xmin", "x1", "left", "bb_left")
    y_min = first_float("y_min", "ymin", "y1", "top", "bb_top")
    if any(name in values for name in ["x_max", "xmax", "x2"]):
        x_max = first_float("x_max", "xmax", "x2")
        y_max = first_float("y_max", "ymax", "y2")
    else:
        width = first_float("width", "w", "bb_width")
        height = first_float("height", "h", "bb_height")
        x_max = x_min + width
        y_max = y_min + height
    timestamp_ns = None
    for name in ["timestamp_ns", "ts", "timestamp", "time"]:
        if name in values:
            parsed = _parse_float_token(values[name])
            if parsed is not None:
                timestamp_ns = int(parsed)
                break
    class_label = values.get("class") or values.get("class_label") or values.get("label")
    return BoundingBox(frame, track_id, x_min, y_min, x_max, y_max, timestamp_ns, class_label)


def _bbox_from_numeric_row(row: list[str], row_index: int) -> BoundingBox | None:
    parsed = [_parse_float_token(token) for token in row]
    numeric = [value for value in parsed if value is not None]
    if len(numeric) < 6:
        return None
    frame = int(numeric[0])
    track_id = int(numeric[1])
    x_min = float(numeric[2])
    y_min = float(numeric[3])
    width = float(numeric[4])
    height = float(numeric[5])
    x_max = x_min + width
    y_max = y_min + height
    return BoundingBox(frame, track_id, x_min, y_min, x_max, y_max, None, None)


def _read_json_tracking_labels(path: Path) -> list[BoundingBox]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations", payload if isinstance(payload, list) else [])
    labels = []
    for index, annotation in enumerate(annotations):
        bbox = annotation.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        x_min, y_min, width, height = [float(value) for value in bbox[:4]]
        frame = int(annotation.get("frame_id", annotation.get("image_id", index)))
        track_id = int(annotation.get("track_id", annotation.get("id", -1)))
        labels.append(
            BoundingBox(
                frame=frame,
                track_id=track_id,
                x_min=x_min,
                y_min=y_min,
                x_max=x_min + width,
                y_max=y_min + height,
                timestamp_ns=annotation.get("timestamp_ns"),
                class_label=str(annotation.get("category_id"))
                if "category_id" in annotation
                else None,
            )
        )
    return labels


def read_tracking_labels(path: str | Path) -> list[BoundingBox]:
    """Read MEVDT tracking labels from COCO JSON, MOT, or headered text/CSV."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _read_json_tracking_labels(path)

    lines = [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return []
    first_row = _split_row(lines[0])
    first_numeric = [_parse_float_token(token) for token in first_row]
    has_header = any(value is None for value in first_numeric)
    labels = []
    header = [token.lower() for token in first_row] if has_header else None
    row_lines = lines[1:] if has_header else lines
    for row_index, line in enumerate(row_lines):
        row = _split_row(line)
        try:
            label = (
                _bbox_from_header_row(header, row, row_index)
                if header is not None
                else _bbox_from_numeric_row(row, row_index)
            )
        except ValueError:
            continue
        if label is not None and label.width > 0.0 and label.height > 0.0:
            labels.append(label)
    return sorted(labels, key=lambda label: (label.track_id, label.frame))


def _events_in_bbox(events: EventBatch, bbox: BoundingBox) -> np.ndarray:
    return (
        (events.x >= bbox.x_min)
        & (events.x <= bbox.x_max)
        & (events.y >= bbox.y_min)
        & (events.y <= bbox.y_max)
    )


def _side_band_counts(events: EventBatch, bbox: BoundingBox, band_fraction: float) -> dict[str, int]:
    if events.count == 0:
        return {edge: 0 for edge in ("left", "right", "top", "bottom")}
    band_x = max(1.0, band_fraction * bbox.width)
    band_y = max(1.0, band_fraction * bbox.height)
    return {
        "left": int(np.sum(events.x <= bbox.x_min + band_x)),
        "right": int(np.sum(events.x >= bbox.x_max - band_x)),
        "top": int(np.sum(events.y <= bbox.y_min + band_y)),
        "bottom": int(np.sum(events.y >= bbox.y_max - band_y)),
    }


def _active_inactive_fractions(
    side_counts: dict[str, int],
    velocity: np.ndarray,
) -> tuple[float, float]:
    total_side_events = float(sum(side_counts.values()))
    if total_side_events <= 0.0:
        return 0.0, 0.0
    if abs(float(velocity[0])) >= abs(float(velocity[1])):
        active_edges = ("left", "right")
        inactive_edges = ("top", "bottom")
    else:
        active_edges = ("top", "bottom")
        inactive_edges = ("left", "right")
    active = sum(side_counts[edge] for edge in active_edges) / total_side_events
    inactive = sum(side_counts[edge] for edge in inactive_edges) / total_side_events
    return float(active), float(inactive)


def compute_bbox_event_diagnostics(
    labels: Iterable[BoundingBox],
    events: EventBatch,
    band_fraction: float = 0.15,
) -> list[TrackWindowDiagnostics]:
    """Compute track-window diagnostics from labels and matching event batches."""
    labels_by_track: dict[int, list[BoundingBox]] = {}
    for label in labels:
        labels_by_track.setdefault(label.track_id, []).append(label)

    diagnostics = []
    for track_id, track_labels in labels_by_track.items():
        track_labels = sorted(track_labels, key=lambda label: label.frame)
        for current, following in zip(track_labels[:-1], track_labels[1:], strict=False):
            if current.timestamp_ns is not None and following.timestamp_ns is not None:
                time_mask = (events.ts >= current.timestamp_ns) & (
                    events.ts < following.timestamp_ns
                )
            else:
                time_mask = np.ones(events.count, dtype=bool)
            bbox_mask = _events_in_bbox(events, current)
            mask = time_mask & bbox_mask
            window_events = EventBatch(
                ts=events.ts[mask],
                x=events.x[mask],
                y=events.y[mask],
                p=events.p[mask],
            )
            velocity = following.center - current.center
            side_counts = _side_band_counts(window_events, current, band_fraction)
            active_fraction, inactive_fraction = _active_inactive_fractions(
                side_counts,
                velocity,
            )

            width_ratio = height_ratio = area_ratio = None
            if window_events.count > 0:
                event_width = float(np.max(window_events.x) - np.min(window_events.x) + 1)
                event_height = float(np.max(window_events.y) - np.min(window_events.y) + 1)
                width_ratio = event_width / current.width
                height_ratio = event_height / current.height
                area_ratio = (event_width * event_height) / current.area

            diagnostics.append(
                TrackWindowDiagnostics(
                    track_id=track_id,
                    frame=current.frame,
                    next_frame=following.frame,
                    timestamp_ns=current.timestamp_ns,
                    next_timestamp_ns=following.timestamp_ns,
                    bbox={
                        "x_min": current.x_min,
                        "y_min": current.y_min,
                        "x_max": current.x_max,
                        "y_max": current.y_max,
                    },
                    center_velocity_px_per_frame=velocity.tolist(),
                    event_count=window_events.count,
                    side_band_counts=side_counts,
                    active_side_fraction=active_fraction,
                    inactive_side_fraction=inactive_fraction,
                    event_bbox_width_ratio=width_ratio,
                    event_bbox_height_ratio=height_ratio,
                    event_bbox_area_ratio=area_ratio,
                )
            )
    return diagnostics


def summarize_diagnostics(diagnostics: Iterable[TrackWindowDiagnostics]) -> dict[str, float | int]:
    """Aggregate MEVDT diagnostics into compact paper-side metrics."""
    diagnostics = list(diagnostics)
    nonempty = [item for item in diagnostics if item.event_count > 0]

    def mean_optional(values):
        values = [value for value in values if value is not None and math.isfinite(value)]
        return float(np.mean(values)) if values else None

    return {
        "windows": len(diagnostics),
        "nonempty_windows": len(nonempty),
        "mean_event_count": float(np.mean([item.event_count for item in nonempty]))
        if nonempty
        else 0.0,
        "mean_active_side_fraction": float(
            np.mean([item.active_side_fraction for item in nonempty])
        )
        if nonempty
        else 0.0,
        "mean_inactive_side_fraction": float(
            np.mean([item.inactive_side_fraction for item in nonempty])
        )
        if nonempty
        else 0.0,
        "mean_event_bbox_width_ratio": mean_optional(
            [item.event_bbox_width_ratio for item in nonempty]
        ),
        "mean_event_bbox_height_ratio": mean_optional(
            [item.event_bbox_height_ratio for item in nonempty]
        ),
        "mean_event_bbox_area_ratio": mean_optional(
            [item.event_bbox_area_ratio for item in nonempty]
        ),
    }
