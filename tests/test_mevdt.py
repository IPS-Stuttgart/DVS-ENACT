import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from dvs_enact import (
    BoundingBox,
    compute_bbox_event_diagnostics,
    read_event_csv,
    read_tracking_labels,
    summarize_diagnostics,
)


def _fixture_file(name, content):
    fixture_dir = Path("outputs") / "test_mevdt"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    path = fixture_dir / f"{uuid4().hex}_{name}"
    path.write_text(content, encoding="utf-8")
    return path


def test_read_event_csv_filters_time_and_bbox():
    event_file = _fixture_file(
        "events.csv",
        "ts,x,y,p\n"
        "10,1,1,1\n"
        "20,5,5,0\n"
        "30,20,20,1\n",
    )
    bbox = BoundingBox(0, 1, 0.0, 0.0, 10.0, 10.0)

    events = read_event_csv(event_file, start_ns=10, end_ns=30, bbox=bbox)

    assert events.count == 2
    assert events.ts.tolist() == [10, 20]
    assert events.x.tolist() == [1, 5]
    assert events.p.tolist() == [1, 0]


def test_read_tracking_labels_supports_headered_xyxy():
    label_file = _fixture_file(
        "labels.csv",
        "frame,track_id,x_min,y_min,x_max,y_max,timestamp_ns,class_label\n"
        "0,7,10,20,30,50,1000,car\n",
    )

    labels = read_tracking_labels(label_file)

    assert len(labels) == 1
    assert labels[0].track_id == 7
    assert labels[0].width == 20.0
    assert labels[0].height == 30.0
    assert labels[0].timestamp_ns == 1000
    assert labels[0].class_label == "car"


def test_read_tracking_labels_supports_mot_rows():
    label_file = _fixture_file("mot.txt", "1,3,10,20,30,40,1,-1,-1\n")

    labels = read_tracking_labels(label_file)

    assert len(labels) == 1
    assert labels[0].frame == 1
    assert labels[0].track_id == 3
    assert labels[0].x_max == 40.0
    assert labels[0].y_max == 60.0


def test_read_tracking_labels_supports_coco_json():
    label_file = _fixture_file(
        "labels.json",
        json.dumps(
            {
                "annotations": [
                    {
                        "image_id": 4,
                        "track_id": 9,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "category_id": 2,
                    }
                ]
            }
        ),
    )

    labels = read_tracking_labels(label_file)

    assert labels[0].frame == 4
    assert labels[0].track_id == 9
    assert labels[0].x_max == 4.0
    assert labels[0].y_max == 6.0


def test_compute_bbox_event_diagnostics_reports_side_support():
    labels = [
        BoundingBox(0, 1, 0.0, 0.0, 10.0, 10.0),
        BoundingBox(1, 1, 2.0, 0.0, 12.0, 10.0),
    ]
    events = read_event_csv_from_text(
        "1,0,5,1\n"
        "2,1,5,1\n"
        "3,9,5,1\n"
        "4,10,5,1\n"
        "5,5,0,1\n"
    )

    diagnostics = compute_bbox_event_diagnostics(labels, events, band_fraction=0.2)
    summary = summarize_diagnostics(diagnostics)

    assert len(diagnostics) == 1
    assert diagnostics[0].side_band_counts["left"] == 2
    assert diagnostics[0].side_band_counts["right"] == 2
    assert diagnostics[0].side_band_counts["top"] == 1
    assert diagnostics[0].active_side_fraction > diagnostics[0].inactive_side_fraction
    assert summary["windows"] == 1
    assert summary["nonempty_windows"] == 1


def read_event_csv_from_text(text):
    from dvs_enact import EventBatch

    rows = [
        [int(value) for value in line.split(",")]
        for line in text.strip().splitlines()
    ]
    return EventBatch(
        ts=np.array([row[0] for row in rows], dtype=np.int64),
        x=np.array([row[1] for row in rows], dtype=np.int32),
        y=np.array([row[2] for row in rows], dtype=np.int32),
        p=np.array([row[3] for row in rows], dtype=np.int8),
    )
