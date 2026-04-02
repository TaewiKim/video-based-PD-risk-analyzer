from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.webapp.stroke_gait_analysis import (  # noqa: E402
    DEFAULT_STROKE_GAIT_THRESHOLDS,
    VOISARD_HS_CALIBRATED_THRESHOLDS,
    build_stroke_gait_analysis,
)
from server.webapp.stroke_gait_dataset_adapters import load_voisard_2025_records  # noqa: E402


def _evaluate(records: list[dict[str, object]], thresholds: dict[str, object], *, cohorts: tuple[str, str]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {cohorts[0]: [], cohorts[1]: []}
    for record in records:
        metadata = record.get("metadata", {})
        cohort = str(metadata.get("cohort", ""))
        if cohort not in grouped:
            continue
        analysis = build_stroke_gait_analysis(record["summary"], thresholds=thresholds)
        grouped[cohort].append(analysis)

    def _share(items: list[dict[str, object]], level: str) -> float:
        if not items:
            return 0.0
        return round(sum(1 for item in items if item.get("pattern_level") == level) / len(items), 3)

    def _mean_score(items: list[dict[str, object]]) -> float:
        if not items:
            return 0.0
        return round(sum(float(item.get("pattern_score", 0) or 0) for item in items) / len(items), 3)

    left, right = cohorts
    return {
        "cohorts": {
            left: {
                "n": len(grouped[left]),
                "mean_pattern_score": _mean_score(grouped[left]),
                "high_share": _share(grouped[left], "High"),
                "moderate_or_higher_share": round(
                    sum(1 for item in grouped[left] if item.get("pattern_level") in {"Moderate", "High"}) / len(grouped[left]),
                    3,
                ) if grouped[left] else 0.0,
            },
            right: {
                "n": len(grouped[right]),
                "mean_pattern_score": _mean_score(grouped[right]),
                "high_share": _share(grouped[right], "High"),
                "moderate_or_higher_share": round(
                    sum(1 for item in grouped[right] if item.get("pattern_level") in {"Moderate", "High"}) / len(grouped[right]),
                    3,
                ) if grouped[right] else 0.0,
            },
        },
        "deltas": {
            "mean_pattern_score": round(_mean_score(grouped[left]) - _mean_score(grouped[right]), 3),
            "high_share": round(_share(grouped[left], "High") - _share(grouped[right], "High"), 3),
        },
    }


def _candidate_thresholds() -> list[tuple[str, dict[str, object]]]:
    candidates: list[tuple[str, dict[str, object]]] = []
    for speed_mod, cadence_mod, stability_mod, variability_mod, tug_mod, turn_mod, visual_mod, high_min in product(
        (0.76, 0.8),
        (96.0, 100.0),
        (0.18, 0.2),
        (5.5, 6.0),
        (14.0, 15.0),
        (2.8, 3.0),
        (1.5, 2.0),
        (5, 6),
    ):
        profile = {
            "speed": {"moderate": speed_mod, "high": 0.35},
            "cadence": {"moderate": cadence_mod, "high": 84.0},
            "asymmetry": {"moderate": 0.22, "high": 0.38},
            "stability": {"moderate": stability_mod, "high": 0.06},
            "tug_seconds": {"moderate": tug_mod, "high": tug_mod + 6.0},
            "turn_duration_seconds": {"moderate": turn_mod, "high": turn_mod + 1.2},
            "visual_gait_assessment": {"moderate": visual_mod, "high": visual_mod + 1.0},
            "stride_time_cv": {"moderate": variability_mod, "high": 9.5},
            "step_time_asymmetry": {"moderate": 0.06, "high": 0.09},
            "pattern_level": {"moderate_min": 2, "high_min": high_min},
        }
        name = (
            f"s{speed_mod:.2f}_c{cadence_mod:.0f}_st{stability_mod:.2f}_"
            f"v{variability_mod:.1f}_t{tug_mod:.0f}_u{turn_mod:.1f}_g{visual_mod:.1f}_h{high_min}"
        )
        candidates.append((name, profile))
    return candidates


def _select_profiles(
    records: list[dict[str, object]],
    *,
    hs_high_share_max: float,
) -> dict[str, object]:
    evaluations: list[dict[str, object]] = []
    for name, thresholds in _candidate_thresholds():
        comparison = _evaluate(records, thresholds, cohorts=("CVA", "HS"))
        hs_high = float(comparison["cohorts"]["HS"]["high_share"])
        cva_high = float(comparison["cohorts"]["CVA"]["high_share"])
        delta_high = float(comparison["deltas"]["high_share"])
        mean_delta = float(comparison["deltas"]["mean_pattern_score"])
        evaluations.append(
            {
                "name": name,
                "thresholds": thresholds,
                "comparison": comparison,
                "ranking": {
                    "hs_high_share": hs_high,
                    "cva_high_share": cva_high,
                    "high_share_delta": delta_high,
                    "mean_score_delta": mean_delta,
                },
            }
        )

    eligible = [
        item
        for item in evaluations
        if float(item["ranking"]["hs_high_share"]) <= hs_high_share_max
    ]
    ranked_pool = eligible or evaluations
    best = sorted(
        ranked_pool,
        key=lambda item: (
            -float(item["ranking"]["high_share_delta"]),
            -float(item["ranking"]["cva_high_share"]),
            float(item["ranking"]["hs_high_share"]),
            -float(item["ranking"]["mean_score_delta"]),
        ),
    )[0]
    frontier = sorted(
        ranked_pool,
        key=lambda item: (
            float(item["ranking"]["hs_high_share"]),
            -float(item["ranking"]["cva_high_share"]),
            -float(item["ranking"]["high_share_delta"]),
        ),
    )[:10]
    return {
        "hs_high_share_max": hs_high_share_max,
        "n_candidates": len(evaluations),
        "n_eligible": len(eligible),
        "best_profile": best,
        "top_frontier": frontier,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare stroke gait threshold profiles on Voisard HS vs CVA cohorts."
    )
    parser.add_argument("voisard_root", help="Path to Voisard unpacked root")
    parser.add_argument("output", nargs="?", help="Optional output JSON path")
    parser.add_argument(
        "--hs-high-share-max",
        type=float,
        default=0.05,
        help="Maximum allowed HS High share for candidate selection. Default: 0.05",
    )
    args = parser.parse_args(argv[1:])

    voisard_root = Path(args.voisard_root)
    output_path = Path(args.output) if args.output else None

    filtered = load_voisard_2025_records(voisard_root, cohort_filter={"CVA", "HS"})
    sweep = _select_profiles(filtered, hs_high_share_max=args.hs_high_share_max)
    recommended_profile_name = str(sweep["best_profile"]["name"])
    recommended_thresholds = sweep["best_profile"]["thresholds"]

    report = {
        "dataset": str(voisard_root),
        "n_records": len(filtered),
        "profiles": {
            "default": _evaluate(filtered, DEFAULT_STROKE_GAIT_THRESHOLDS, cohorts=("CVA", "HS")),
            "voisard_hs_calibrated": _evaluate(filtered, VOISARD_HS_CALIBRATED_THRESHOLDS, cohorts=("CVA", "HS")),
            recommended_profile_name: _evaluate(filtered, recommended_thresholds, cohorts=("CVA", "HS")),
        },
        "recommended_profile": recommended_profile_name,
        "threshold_sweep": sweep,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output_path:
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
