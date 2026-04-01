from __future__ import annotations

from server.webapp.stroke_gait_analysis import build_stroke_gait_analysis


def test_build_stroke_gait_analysis_handles_missing_summary() -> None:
    payload = build_stroke_gait_analysis(None)
    assert payload["available"] is False
    assert payload["pattern_score"] == 0
    assert payload["pattern_level"] == "Unknown"


def test_build_stroke_gait_analysis_low_pattern_case() -> None:
    payload = build_stroke_gait_analysis(
        {
            "avg_speed": 1.02,
            "avg_cadence": 108.0,
            "avg_asymmetry": 0.08,
            "avg_stability": 0.31,
            "avg_stride_time_cv": 2.8,
            "avg_step_time_asymmetry": 0.03,
        }
    )
    assert payload["available"] is True
    assert payload["pattern_level"] == "Low"
    assert payload["pattern_score"] <= 1
    assert payload["speed_band"]["label"] == "community_plus"


def test_build_stroke_gait_analysis_high_pattern_case() -> None:
    payload = build_stroke_gait_analysis(
        {
            "avg_speed": 0.34,
            "avg_cadence": 82.0,
            "avg_asymmetry": 0.42,
            "avg_stability": 0.05,
            "avg_stride_time_cv": 9.6,
            "avg_step_time_asymmetry": 0.09,
        }
    )
    assert payload["pattern_level"] == "High"
    assert payload["pattern_score"] >= 5
    assert payload["flagged_count"] >= 4
    assert payload["domain_scores"]["walking_capacity"]["score"] == 2
    assert payload["domain_scores"]["spatial_asymmetry"]["score"] == 2
    assert payload["speed_band"]["label"] == "household_only"
