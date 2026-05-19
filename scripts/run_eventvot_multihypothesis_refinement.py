"""Command-line entry point for EventVOT multi-hypothesis refinement."""

from __future__ import annotations

import argparse
import json

from eventvot_multihypothesis_core import (  # pylint: disable=import-error
    EventVOTMultiHypothesisConfig,
    run_multihypothesis,
)
from run_eventvot_refinement import (  # pylint: disable=import-error
    EventVOTRefinementOptions,
    _acceptance_config_from_args,
    _refiner_from_args,
    _resolve_cli_config_tracker_path,
    _resolve_cli_output_results,
    build_parser as build_base_parser,
    load_requested_sequence_names,
)


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser()
    parser.description = (
        "Refine EventVOT xywh tracker results with "
        "multi-hypothesis DVS-ENACT."
    )
    parser.add_argument("--disable-multi-hypothesis", action="store_true")
    parser.add_argument("--candidate-refinement-blends", default="0.15,0.25,0.35")
    parser.add_argument(
        "--candidate-search-expansion-factors",
        default="1.15,1.25,1.40",
    )
    parser.add_argument("--candidate-max-events", default="64,128,256")
    parser.add_argument(
        "--candidate-measurement-noise-variances",
        default="2.0,4.0,8.0",
    )
    parser.add_argument("--candidate-event-activity-floors", default="0.03,0.05,0.08")
    parser.add_argument("--candidate-inactive-activity-thresholds", default="")
    parser.add_argument("--include-no-polarity-hypothesis", action="store_true")
    parser.add_argument("--combine-hypothesis-values", action="store_true")
    parser.add_argument("--max-hypotheses-per-frame", type=int, default=24)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_multihypothesis(
        EventVOTRefinementOptions(
            eventvot_root=args.eventvot_root,
            base_results=args.base_results,
            output_results=_resolve_cli_output_results(args),
            split=args.split,
            sequences=load_requested_sequence_names(
                args.sequence,
                args.sequence_list,
                args.sequence_file,
            ),
            sequence_index=args.sequence_index,
            sequence_count=args.sequence_count,
            tracker_name=args.tracker_name,
            skip_existing=not args.no_skip_existing,
            event_column_order=args.event_column_order,
            diagnostics_json=args.diagnostics_json,
            config_tracker_path=_resolve_cli_config_tracker_path(args),
            acceptance_config=_acceptance_config_from_args(args),
        ),
        refiner=_refiner_from_args(args),
        multi_hypothesis_config=_multihypothesis_config_from_args(args),
    )
    print(json.dumps(payload["summary"], indent=2))
    return 0


def _multihypothesis_config_from_args(
    args: argparse.Namespace,
) -> EventVOTMultiHypothesisConfig:
    return EventVOTMultiHypothesisConfig(
        enabled=not args.disable_multi_hypothesis,
        refinement_blends=_parse_float_tuple(args.candidate_refinement_blends),
        search_expansion_factors=_parse_float_tuple(
            args.candidate_search_expansion_factors,
        ),
        max_events=_parse_int_or_none_tuple(args.candidate_max_events),
        measurement_noise_variances=_parse_float_tuple(
            args.candidate_measurement_noise_variances,
        ),
        event_activity_floors=_parse_float_tuple(args.candidate_event_activity_floors),
        inactive_activity_thresholds=_parse_float_tuple(
            args.candidate_inactive_activity_thresholds,
        ),
        include_without_event_polarity=args.include_no_polarity_hypothesis,
        combine_values=args.combine_hypothesis_values,
        max_hypotheses_per_frame=args.max_hypotheses_per_frame,
    )


def _parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(token) for token in _split_values(text))


def _parse_int_or_none_tuple(text: str) -> tuple[int | None, ...]:
    return tuple(
        None if token.lower() in {"none", "unlimited"} else int(token)
        for token in _split_values(text)
    )


def _split_values(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.replace(",", " ").split() if token)


if __name__ == "__main__":
    raise SystemExit(main())
