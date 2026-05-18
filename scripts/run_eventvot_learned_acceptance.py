"""Train and replay validation-learned EventVOT acceptance policies.

This utility turns an existing DVS-ENACT diagnostics JSON into a lightweight
accept/reject model for post-hoc EventVOT refinement.  The expensive refiner run
is reused: validation ground truth labels whether each stored refinement improves
its base tracker box, a small logistic policy is fitted from diagnostic features,
and the learned policy is replayed to write ordinary EventVOT result files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_eventvot_acceptance_replay import (  # noqa: E402
    ReplayAcceptanceConfig,
    evaluate_frame_acceptance,
    frame_refiner_output_xywh,
    resolve_replay_base_result_file,
    select_sequence_summaries,
    write_decisions_csv,
)
from run_eventvot_refinement import (  # noqa: E402
    box_iou_xywh,
    load_xywh_result_file,
    resolve_eventvot_split_root,
    save_xywh_result_file,
)
from run_eventvot_validation_sweep import (  # noqa: E402
    evaluate_eventvot_results,
    load_numeric_matrix,
)

RAW_FEATURE_NAMES = (
    "log1p_event_count",
    "log1p_used_event_count",
    "log1p_active_measurement_count",
    "mean_event_activity",
    "candidate_iou",
    "log_candidate_area_ratio",
    "center_shift_ratio",
    "raw_candidate_iou",
    "log_raw_candidate_area_ratio",
    "raw_center_shift_ratio",
    "polarity_consistency_fraction",
    "mean_event_polarity_weight",
    "log1p_quadratic_form_per_active_measurement",
    "active_fraction",
)
POLICY_SCHEMA_VERSION = 1
EPSILON = 1.0e-12


@dataclass(frozen=True)
class LearnedAcceptanceOptions:
    """Inputs and outputs for learned acceptance policy training/replay."""

    diagnostics_json: Path
    output_results: Path
    eventvot_root: Path | None = None
    base_results: Path | None = None
    split: str | None = None
    sequences: tuple[str, ...] = ()
    policy_json: Path | None = None
    load_policy_json: Path | None = None
    summary_json: Path | None = None
    decisions_csv: Path | None = None
    skip_evaluation: bool = False
    min_iou_gain: float = 0.0
    l2: float = 1.0e-2
    learning_rate: float = 5.0e-2
    epochs: int = 2000
    threshold_count: int = 51
    objective: str = "sr_auc"


def main() -> int:
    options = options_from_args(build_parser().parse_args())
    payload = run(options)
    print(json.dumps(payload["summary"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a validation-learned EventVOT DVS-ENACT acceptance policy "
            "from diagnostics and replay it into result files."
        )
    )
    parser.add_argument("--diagnostics-json", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument(
        "--eventvot-root",
        type=Path,
        help="Required for training and for evaluation unless --skip-evaluation is used.",
    )
    parser.add_argument("--base-results", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument(
        "--policy-json",
        type=Path,
        help="Where to write the trained policy. Defaults to <output-results>/learned_policy.json.",
    )
    parser.add_argument(
        "--load-policy-json",
        type=Path,
        help="Skip training and replay an existing learned policy JSON.",
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--decisions-csv", type=Path)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Write replayed result files without computing EventVOT metrics.",
    )
    parser.add_argument(
        "--min-iou-gain",
        type=float,
        default=0.0,
        help=(
            "Validation label margin: a refinement is positive only when its "
            "IoU exceeds the base IoU by more than this value."
        ),
    )
    parser.add_argument("--l2", type=float, default=1.0e-2)
    parser.add_argument("--learning-rate", type=float, default=5.0e-2)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--threshold-count",
        type=int,
        default=51,
        help="Number of probability thresholds to evaluate during validation tuning.",
    )
    parser.add_argument(
        "--objective",
        choices=("sr_auc", "pr_auc", "pr_20", "npr_auc", "npr_020", "mean_iou"),
        default="sr_auc",
        help="EventVOT metric used to pick the replay probability threshold.",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> LearnedAcceptanceOptions:
    return LearnedAcceptanceOptions(
        diagnostics_json=args.diagnostics_json,
        output_results=args.output_results,
        eventvot_root=args.eventvot_root,
        base_results=args.base_results,
        split=args.split,
        sequences=tuple(args.sequence),
        policy_json=args.policy_json,
        load_policy_json=args.load_policy_json,
        summary_json=args.summary_json,
        decisions_csv=args.decisions_csv,
        skip_evaluation=args.skip_evaluation,
        min_iou_gain=args.min_iou_gain,
        l2=args.l2,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        threshold_count=args.threshold_count,
        objective=args.objective,
    )


def run(options: LearnedAcceptanceOptions) -> dict[str, Any]:
    """Train/load a policy, replay it, and optionally evaluate EventVOT metrics."""

    diagnostics = json.loads(options.diagnostics_json.read_text(encoding="utf-8"))
    split = str(options.split or diagnostics.get("options", {}).get("split", "val"))
    selected_summaries = select_sequence_summaries(
        diagnostics.get("sequences", []),
        options.sequences,
    )
    if not selected_summaries:
        raise ValueError("No sequence diagnostics selected for learned replay")

    split_root = None
    if options.eventvot_root is not None:
        split_root = resolve_eventvot_split_root(options.eventvot_root, split)

    if options.load_policy_json is not None:
        policy = load_policy(options.load_policy_json)
        training_summary = None
        threshold_rows: list[dict[str, Any]] = []
    else:
        if split_root is None:
            raise ValueError("--eventvot-root is required when training a policy")
        examples, training_summary = collect_training_examples(
            selected_summaries,
            split_root,
            base_results=options.base_results,
            min_iou_gain=options.min_iou_gain,
        )
        policy = fit_learned_policy(examples, options)
        policy, threshold_rows = tune_policy_threshold(
            policy,
            examples,
            selected_summaries,
            split_root,
            base_results=options.base_results,
            threshold_count=options.threshold_count,
            objective=options.objective,
            use_eventvot_metrics=not options.skip_evaluation,
        )
        policy_json = options.policy_json or options.output_results / "learned_policy.json"
        write_policy(policy_json, policy)

    sequence_names = [str(summary["sequence"]) for summary in selected_summaries]
    sequence_outputs, counts, decisions = replay_policy_to_results(
        policy,
        selected_summaries,
        base_results=options.base_results,
        output_results=options.output_results,
    )

    metrics = None
    if not options.skip_evaluation:
        if split_root is None:
            raise ValueError("--eventvot-root is required unless --skip-evaluation is used")
        metrics = evaluate_eventvot_results(split_root, options.output_results, sequence_names)

    if options.decisions_csv is not None:
        write_decisions_csv(options.decisions_csv, decisions)

    summary = {
        "sequence_count": len(sequence_outputs),
        "frame_count": sum(output["frame_count"] for output in sequence_outputs),
        "accepted_refinement_count": int(counts.get("accepted", 0)),
        "acceptance_counts": dict(sorted(counts.items())),
        "policy_threshold": float(policy["threshold"]),
        "metrics": metrics,
        "output_results": str(options.output_results),
    }
    payload = {
        "schema_version": 1,
        "description": (
            "Validation-learned EventVOT DVS-ENACT acceptance replay. The model "
            "scores each stored refinement from diagnostics and replaces the base "
            "tracker box only when the learned probability crosses the selected "
            "validation threshold."
        ),
        "split": split,
        "diagnostics_json": str(options.diagnostics_json),
        "policy": policy,
        "training_summary": training_summary,
        "threshold_sweep": threshold_rows,
        "summary": summary,
        "sequences": sequence_outputs,
    }
    summary_json = options.summary_json or options.output_results / "learned_acceptance_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def collect_training_examples(
    sequence_summaries: list[dict[str, Any]],
    split_root: Path,
    *,
    base_results: Path | None,
    min_iou_gain: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return labeled per-frame examples from diagnostics and validation GT."""

    examples: list[dict[str, Any]] = []
    skipped = Counter()
    for sequence_summary in sequence_summaries:
        sequence_name = str(sequence_summary["sequence"])
        base_result_file = resolve_replay_base_result_file(sequence_summary, base_results)
        base_boxes = load_xywh_result_file(base_result_file)
        groundtruth, evaluable_mask = load_sequence_supervision(
            split_root / sequence_name,
            base_boxes.shape[0],
        )
        frames = sequence_summary.get("frames", [])
        if not frames:
            skipped["missing_frame_diagnostics"] += int(base_boxes.shape[0])
            continue
        for frame in frames:
            frame_index = int(frame.get("frame_index", -1))
            if frame_index <= 0 or frame_index >= base_boxes.shape[0]:
                skipped["initial_or_out_of_range"] += 1
                continue
            if frame_index >= groundtruth.shape[0] or not bool(evaluable_mask[frame_index]):
                skipped["not_evaluable"] += 1
                continue
            if frame.get("fallback_reason") is not None:
                skipped["fallback"] += 1
                continue
            try:
                refined_xywh = frame_refiner_output_xywh(frame)
            except ValueError:
                skipped["missing_refiner_output"] += 1
                continue
            if not finite_xywh(refined_xywh):
                skipped["invalid_refiner_output"] += 1
                continue

            base_xywh = np.asarray(base_boxes[frame_index], dtype=float)
            gt_xywh = np.asarray(groundtruth[frame_index], dtype=float)
            base_iou = box_iou_xywh(base_xywh, gt_xywh)
            refined_iou = box_iou_xywh(refined_xywh, gt_xywh)
            iou_gain = float(refined_iou - base_iou)
            examples.append(
                {
                    "sequence": sequence_name,
                    "frame_index": frame_index,
                    "features": extract_raw_features(base_xywh, frame),
                    "label": bool(iou_gain > min_iou_gain),
                    "base_iou": float(base_iou),
                    "refined_iou": float(refined_iou),
                    "iou_gain": iou_gain,
                }
            )

    if not examples:
        raise ValueError("No trainable examples found in diagnostics/ground truth")
    label_count = Counter(bool(example["label"]) for example in examples)
    summary = {
        "example_count": len(examples),
        "positive_count": int(label_count[True]),
        "negative_count": int(label_count[False]),
        "min_iou_gain": float(min_iou_gain),
        "skipped_counts": dict(sorted(skipped.items())),
    }
    return examples, summary


def load_sequence_supervision(
    sequence_dir: Path,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load GT boxes and the same present-frame mask used by EventVOT eval."""

    groundtruth = load_numeric_matrix(sequence_dir / "groundtruth.txt", min_columns=4)[:, :4]
    absent = load_numeric_matrix(sequence_dir / "absent.txt", min_columns=1)[:, 0]
    usable = min(frame_count, groundtruth.shape[0], absent.shape[0])
    groundtruth = groundtruth[:usable, :]
    absent = absent[:usable]
    official_absent = 1.0 - absent
    present_mask = ~np.isclose(official_absent, 1.0)
    finite_gt = np.all(np.isfinite(groundtruth), axis=1)
    valid_box = (groundtruth[:, 2] > 0.0) & (groundtruth[:, 3] > 0.0)
    valid_box &= np.sum(groundtruth > 0.0, axis=1) == 4
    return groundtruth, present_mask & finite_gt & valid_box


def extract_raw_features(base_xywh: np.ndarray, frame: dict[str, Any]) -> list[float]:
    """Return the fixed diagnostic feature vector for one candidate frame."""

    decision = evaluate_frame_acceptance(
        np.asarray(base_xywh, dtype=float),
        frame,
        ReplayAcceptanceConfig(enabled=False),
    )
    return [
        safe_log1p(frame.get("event_count")),
        safe_log1p(frame.get("used_event_count")),
        safe_log1p(frame.get("active_measurement_count")),
        optional_float(frame.get("mean_event_activity")),
        decision.candidate_iou,
        safe_log_ratio(decision.candidate_area_ratio),
        decision.center_shift_ratio,
        decision.raw_candidate_iou,
        safe_log_ratio(decision.raw_candidate_area_ratio),
        decision.raw_center_shift_ratio,
        optional_float(frame.get("polarity_consistency_fraction")),
        optional_float(frame.get("mean_event_polarity_weight")),
        safe_log1p(decision.quadratic_form_per_active_measurement),
        optional_float(decision.active_fraction),
    ]


def fit_learned_policy(
    examples: list[dict[str, Any]],
    options: LearnedAcceptanceOptions,
) -> dict[str, Any]:
    """Fit a dependency-free logistic accept/reject model."""

    raw_features = np.asarray([example["features"] for example in examples], dtype=float)
    labels = np.asarray([float(example["label"]) for example in examples], dtype=float)
    design, normalizer = make_design_matrix(raw_features)
    weights, bias, training_loss = fit_logistic_regression(
        design,
        labels,
        l2=options.l2,
        learning_rate=options.learning_rate,
        epochs=options.epochs,
    )
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "model": "standardized_logistic_regression_with_missing_indicators",
        "raw_feature_names": list(RAW_FEATURE_NAMES),
        "expanded_feature_names": [
            *RAW_FEATURE_NAMES,
            *(f"{name}_missing" for name in RAW_FEATURE_NAMES),
        ],
        "impute_values": normalizer["impute_values"].astype(float).tolist(),
        "feature_means": normalizer["feature_means"].astype(float).tolist(),
        "feature_scales": normalizer["feature_scales"].astype(float).tolist(),
        "weights": weights.astype(float).tolist(),
        "bias": float(bias),
        "threshold": 0.5,
        "training_loss": float(training_loss),
        "training_options": {
            "min_iou_gain": float(options.min_iou_gain),
            "l2": float(options.l2),
            "learning_rate": float(options.learning_rate),
            "epochs": int(options.epochs),
        },
    }


def tune_policy_threshold(
    policy: dict[str, Any],
    examples: list[dict[str, Any]],
    sequence_summaries: list[dict[str, Any]],
    split_root: Path,
    *,
    base_results: Path | None,
    threshold_count: int,
    objective: str,
    use_eventvot_metrics: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pick a probability threshold using validation labels or EventVOT metrics."""

    scores = np.asarray([score_raw_features(policy, example["features"]) for example in examples])
    thresholds = candidate_thresholds(scores, threshold_count)
    if not use_eventvot_metrics:
        rows = threshold_label_sweep(thresholds, scores, examples)
        best = max(
            rows,
            key=lambda row: (
                row["f1"],
                row["balanced_accuracy"],
                -row["accepted_refinement_count"],
            ),
        )
        selected = dict(policy)
        selected["threshold"] = float(best["threshold"])
        selected["threshold_selection"] = {
            "method": "training_label_f1",
            "selected": best,
        }
        return selected, rows

    rows: list[dict[str, Any]] = []
    sequence_names = [str(summary["sequence"]) for summary in sequence_summaries]
    with tempfile.TemporaryDirectory(prefix="dvsenact_learned_acceptance_") as tmp:
        tmp_dir = Path(tmp)
        for threshold in thresholds:
            candidate_policy = dict(policy)
            candidate_policy["threshold"] = float(threshold)
            _sequence_outputs, counts, _decisions = replay_policy_to_results(
                candidate_policy,
                sequence_summaries,
                base_results=base_results,
                output_results=tmp_dir,
            )
            metrics = evaluate_eventvot_results(split_root, tmp_dir, sequence_names)
            rows.append(
                {
                    "threshold": float(threshold),
                    "accepted_refinement_count": int(counts.get("accepted", 0)),
                    **metrics,
                }
            )
    best = max(rows, key=lambda row: threshold_rank(row, objective))
    selected = dict(policy)
    selected["threshold"] = float(best["threshold"])
    selected["threshold_selection"] = {
        "method": "eventvot_metric",
        "objective": objective,
        "selected": best,
    }
    return selected, rows


def threshold_label_sweep(
    thresholds: np.ndarray,
    scores: np.ndarray,
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = np.asarray([bool(example["label"]) for example in examples], dtype=bool)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predicted = scores >= float(threshold)
        true_positive = int(np.sum(predicted & labels))
        false_positive = int(np.sum(predicted & ~labels))
        false_negative = int(np.sum(~predicted & labels))
        true_negative = int(np.sum(~predicted & ~labels))
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        specificity = safe_divide(true_negative, true_negative + false_positive)
        rows.append(
            {
                "threshold": float(threshold),
                "accepted_refinement_count": int(np.sum(predicted)),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": safe_divide(2.0 * precision * recall, precision + recall),
                "balanced_accuracy": 0.5 * (recall + specificity),
            }
        )
    return rows


def threshold_rank(row: dict[str, Any], objective: str) -> tuple[float, float, float, float, int]:
    return (
        float(row[objective]),
        float(row.get("sr_auc", 0.0)),
        float(row.get("pr_auc", 0.0)),
        float(row.get("npr_auc", 0.0)),
        -int(row.get("accepted_refinement_count", 0)),
    )


def replay_policy_to_results(
    policy: dict[str, Any],
    sequence_summaries: list[dict[str, Any]],
    *,
    base_results: Path | None,
    output_results: Path,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    """Replay a learned policy into EventVOT result files."""

    output_results.mkdir(parents=True, exist_ok=True)
    aggregate_counts: Counter[str] = Counter()
    decisions: list[dict[str, Any]] = []
    sequence_outputs: list[dict[str, Any]] = []
    for sequence_summary in sequence_summaries:
        sequence_name = str(sequence_summary["sequence"])
        base_result_file = resolve_replay_base_result_file(sequence_summary, base_results)
        base_boxes = load_xywh_result_file(base_result_file)
        replayed_boxes = np.asarray(base_boxes, dtype=float).copy()
        sequence_counts: Counter[str] = Counter()

        for frame in sequence_summary.get("frames", []):
            frame_index = int(frame.get("frame_index", -1))
            if frame_index < 0 or frame_index >= base_boxes.shape[0]:
                continue
            if frame_index == 0:
                sequence_counts["initial_frame"] += 1
                continue
            decision = evaluate_learned_acceptance(
                policy,
                np.asarray(base_boxes[frame_index], dtype=float),
                frame,
            )
            reason_key = "accepted" if decision["accepted"] else decision["reason"]
            sequence_counts[reason_key] += 1
            if decision["accepted"]:
                replayed_boxes[frame_index] = frame_refiner_output_xywh(frame)
            decisions.append(
                {
                    "sequence": sequence_name,
                    "frame_index": frame_index,
                    **decision,
                }
            )

        save_xywh_result_file(output_results / f"{sequence_name}.txt", replayed_boxes)
        aggregate_counts.update(sequence_counts)
        sequence_outputs.append(
            {
                "sequence": sequence_name,
                "base_result_file": str(base_result_file),
                "output_result_file": str(output_results / f"{sequence_name}.txt"),
                "frame_count": int(base_boxes.shape[0]),
                "acceptance_counts": dict(sorted(sequence_counts.items())),
            }
        )
    return sequence_outputs, aggregate_counts, decisions


def evaluate_learned_acceptance(
    policy: dict[str, Any],
    base_xywh: np.ndarray,
    frame: dict[str, Any],
) -> dict[str, Any]:
    """Return one learned accept/reject decision from stored diagnostics."""

    fallback_reason = frame.get("fallback_reason")
    if fallback_reason is not None:
        return {
            "accepted": False,
            "reason": f"fallback:{fallback_reason}",
            "probability": 0.0,
            "threshold": float(policy["threshold"]),
        }
    try:
        refiner_output = frame_refiner_output_xywh(frame)
    except ValueError:
        return {
            "accepted": False,
            "reason": "missing_refiner_output",
            "probability": 0.0,
            "threshold": float(policy["threshold"]),
        }
    if not finite_xywh(refiner_output):
        return {
            "accepted": False,
            "reason": "invalid_refiner_output",
            "probability": 0.0,
            "threshold": float(policy["threshold"]),
        }

    features = extract_raw_features(base_xywh, frame)
    probability = score_raw_features(policy, features)
    accepted = bool(probability >= float(policy["threshold"]))
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "learned_probability",
        "probability": float(probability),
        "threshold": float(policy["threshold"]),
    }


def make_design_matrix(
    raw_features: np.ndarray,
    normalizer: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = np.asarray(raw_features, dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    finite = np.isfinite(raw)
    if normalizer is None:
        masked = np.where(finite, raw, np.nan)
        impute_values = np.nanmedian(masked, axis=0)
        impute_values = np.where(np.isfinite(impute_values), impute_values, 0.0)
        filled = np.where(finite, raw, impute_values)
        feature_means = np.mean(filled, axis=0)
        feature_scales = np.std(filled, axis=0)
        feature_scales = np.where(feature_scales > EPSILON, feature_scales, 1.0)
        normalizer = {
            "impute_values": impute_values,
            "feature_means": feature_means,
            "feature_scales": feature_scales,
        }
    else:
        impute_values = normalizer["impute_values"]
        feature_means = normalizer["feature_means"]
        feature_scales = normalizer["feature_scales"]
        filled = np.where(finite, raw, impute_values)

    standardized = (filled - feature_means) / feature_scales
    missing_flags = (~finite).astype(float)
    return np.hstack((standardized, missing_flags)), normalizer


def fit_logistic_regression(
    design: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    learning_rate: float,
    epochs: int,
) -> tuple[np.ndarray, float, float]:
    """Fit a small balanced logistic model with deterministic gradient descent."""

    design = np.asarray(design, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if design.ndim != 2:
        raise ValueError("design matrix must be two-dimensional")
    if design.shape[0] != labels.shape[0]:
        raise ValueError("labels length does not match design matrix")

    positive_fraction = float(np.mean(labels)) if labels.size else 0.0
    bias = logit(min(max(positive_fraction, EPSILON), 1.0 - EPSILON))
    weights = np.zeros(design.shape[1], dtype=float)
    if labels.size == 0 or positive_fraction in {0.0, 1.0}:
        probabilities = sigmoid(design @ weights + bias)
        return weights, bias, logistic_loss(probabilities, labels, weights, l2)

    positive_weight = 0.5 / positive_fraction
    negative_weight = 0.5 / (1.0 - positive_fraction)
    sample_weights = np.where(labels > 0.5, positive_weight, negative_weight)
    weight_sum = float(np.sum(sample_weights))
    for _epoch in range(max(1, int(epochs))):
        probabilities = sigmoid(design @ weights + bias)
        residual = (probabilities - labels) * sample_weights
        grad_weights = design.T @ residual / weight_sum + float(l2) * weights
        grad_bias = float(np.sum(residual) / weight_sum)
        weights -= float(learning_rate) * grad_weights
        bias -= float(learning_rate) * grad_bias

    probabilities = sigmoid(design @ weights + bias)
    return weights, bias, logistic_loss(probabilities, labels, weights, l2)


def logistic_loss(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    cross_entropy = -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return float(cross_entropy + 0.5 * float(l2) * float(np.sum(weights**2)))


def score_raw_features(policy: dict[str, Any], features: list[float]) -> float:
    raw = np.asarray(features, dtype=float).reshape(1, -1)
    normalizer = {
        "impute_values": np.asarray(policy["impute_values"], dtype=float),
        "feature_means": np.asarray(policy["feature_means"], dtype=float),
        "feature_scales": np.asarray(policy["feature_scales"], dtype=float),
    }
    design, _normalizer = make_design_matrix(raw, normalizer)
    weights = np.asarray(policy["weights"], dtype=float)
    score = float(design @ weights + float(policy["bias"]))
    return float(sigmoid(score))


def candidate_thresholds(scores: np.ndarray, threshold_count: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.asarray([1.0], dtype=float)
    quantiles = np.linspace(0.0, 1.0, max(2, int(threshold_count)))
    values = np.quantile(scores, quantiles)
    candidates = np.concatenate((np.asarray([0.0, 0.5, 1.0]), values))
    candidates = np.clip(candidates, 0.0, 1.0)
    return np.asarray(sorted({round(float(value), 12) for value in candidates}), dtype=float)


def write_policy(path: Path, policy: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if int(policy.get("schema_version", -1)) != POLICY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported learned policy schema in {path}")
    if tuple(policy.get("raw_feature_names", ())) != RAW_FEATURE_NAMES:
        raise ValueError(f"Learned policy features do not match this script: {path}")
    return policy


def finite_xywh(box: np.ndarray) -> bool:
    values = np.asarray(box, dtype=float)
    return values.shape == (4,) and np.all(np.isfinite(values)) and bool(np.all(values[2:] > 0.0))


def safe_log1p(value: Any) -> float:
    numeric = optional_float(value)
    if numeric is None or numeric < 0.0:
        return math.nan
    return float(math.log1p(numeric))


def safe_log_ratio(value: Any) -> float:
    numeric = optional_float(value)
    if numeric is None or numeric <= 0.0:
        return math.nan
    return float(math.log(numeric))


def optional_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(value, dtype=float)
    positive = values >= 0.0
    negative = ~positive
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    return float(result) if np.isscalar(value) else result


def logit(probability: float) -> float:
    probability = min(max(float(probability), EPSILON), 1.0 - EPSILON)
    return float(math.log(probability / (1.0 - probability)))


if __name__ == "__main__":
    raise SystemExit(main())
