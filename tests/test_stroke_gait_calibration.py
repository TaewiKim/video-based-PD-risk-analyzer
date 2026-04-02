from __future__ import annotations

from server.webapp.stroke_gait_calibration import (
    build_group_comparison,
    build_calibration_record,
    build_calibration_record_from_summary,
    build_calibration_report,
    filter_calibration_records,
)


def test_build_calibration_record_extracts_nested_summary() -> None:
    payload = {
        "gait_analysis": {
            "summary": {
                "avg_speed": 0.55,
                "avg_cadence": 96.0,
                "avg_asymmetry": 0.18,
                "avg_stability": 0.14,
                "avg_stride_time_cv": 5.1,
                "avg_step_time_asymmetry": 0.04,
            }
        }
    }
    record = build_calibration_record("sample.json", payload)
    assert record["file"] == "sample.json"
    assert record["pattern_level"] in {"Low", "Moderate", "High"}
    assert "stroke_gait_analysis" in record


def test_build_calibration_report_aggregates_counts() -> None:
    records = [
        {
            "file": "a.json",
            "pattern_level": "High",
            "pattern_score": 6,
            "speed_band": "household_only",
            "flagged_count": 4,
            "stroke_gait_analysis": {
                "summary_note": "A",
                "domain_scores": {
                    "walking_capacity": {"score": 2},
                    "spatial_asymmetry": {"score": 2},
                },
            },
        },
        {
            "file": "b.json",
            "pattern_level": "Moderate",
            "pattern_score": 3,
            "speed_band": "limited_community",
            "flagged_count": 2,
            "stroke_gait_analysis": {
                "summary_note": "B",
                "domain_scores": {
                    "walking_capacity": {"score": 1},
                    "temporal_variability": {"score": 1},
                },
            },
        },
    ]
    report = build_calibration_report(records)
    assert report["n_records"] == 2
    assert report["pattern_level_counts"]["High"] == 1
    assert report["speed_band_counts"]["household_only"] == 1
    assert report["threshold_review"]["prominent_pattern_share"] == 0.5
    assert report["domain_flag_counts"]["walking_capacity"] == 2
    assert report["cohort_counts"] == {}
    assert report["label_counts"] == {}


def test_build_calibration_record_from_summary_keeps_metadata() -> None:
    summary = {
        "avg_speed": 0.42,
        "avg_cadence": 92.0,
        "avg_asymmetry": 0.21,
        "avg_stability": 0.16,
        "avg_stride_time_cv": 4.2,
        "avg_step_time_asymmetry": 0.03,
    }
    record = build_calibration_record_from_summary(
        "external-row-1",
        summary,
        metadata={"subject_id": "P01", "cohort": "stroke"},
    )
    assert record["file"] == "external-row-1"
    assert record["metadata"]["subject_id"] == "P01"
    assert record["stroke_gait_analysis"]["available"] is True


def test_build_calibration_report_tracks_metadata_breakdown() -> None:
    report = build_calibration_report(
        [
            {
                "file": "x.json",
                "pattern_level": "High",
                "pattern_score": 5,
                "speed_band": "limited_community",
                "flagged_count": 3,
                "metadata": {"cohort": "stroke", "label": "case"},
                "stroke_gait_analysis": {"summary_note": "X", "domain_scores": {}},
            },
            {
                "file": "y.json",
                "pattern_level": "Low",
                "pattern_score": 0,
                "speed_band": "community_plus",
                "flagged_count": 0,
                "metadata": {"cohort": "control", "label": "control"},
                "stroke_gait_analysis": {"summary_note": "Y", "domain_scores": {}},
            },
        ]
    )
    assert report["cohort_counts"] == {"stroke": 1, "control": 1}
    assert report["label_counts"] == {"case": 1, "control": 1}


def test_filter_calibration_records_filters_by_cohort() -> None:
    records = [
        {"metadata": {"cohort": "CVA", "label": "stroke"}},
        {"metadata": {"cohort": "HS", "label": "healthy"}},
    ]
    filtered = filter_calibration_records(records, cohort_in={"CVA"})
    assert len(filtered) == 1
    assert filtered[0]["metadata"]["cohort"] == "CVA"


def test_build_group_comparison_compares_two_cohorts() -> None:
    comparison = build_group_comparison(
        [
            {"pattern_score": 5, "pattern_level": "High", "metadata": {"cohort": "CVA"}},
            {"pattern_score": 3, "pattern_level": "Moderate", "metadata": {"cohort": "CVA"}},
            {"pattern_score": 1, "pattern_level": "Low", "metadata": {"cohort": "HS"}},
            {"pattern_score": 2, "pattern_level": "Moderate", "metadata": {"cohort": "HS"}},
        ],
        groups=("CVA", "HS"),
    )
    assert comparison["groups"]["CVA"]["n_records"] == 2
    assert comparison["groups"]["HS"]["mean_pattern_score"] == 1.5
    assert comparison["deltas"]["mean_pattern_score"] == 2.5
