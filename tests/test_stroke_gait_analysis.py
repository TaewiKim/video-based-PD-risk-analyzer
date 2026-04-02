from __future__ import annotations

from server.webapp.stroke_gait_analysis import (
    VOISARD_HS_CALIBRATED_THRESHOLDS,
    build_stroke_gait_layers,
    build_stroke_gait_analysis,
)


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


def test_build_stroke_gait_analysis_supports_threshold_override() -> None:
    payload = build_stroke_gait_analysis(
        {
            "avg_speed": 0.74,
            "avg_cadence": 97.0,
            "avg_asymmetry": 0.19,
            "avg_stability": 0.17,
            "avg_stride_time_cv": 4.8,
            "avg_step_time_asymmetry": 0.05,
        },
        thresholds=VOISARD_HS_CALIBRATED_THRESHOLDS,
    )
    assert payload["speed_band"]["community_threshold_mps"] == 0.72
    assert payload["threshold_profile"]["pattern_level"]["high_min"] == 6


def test_build_stroke_gait_analysis_uses_meta_mobility_features() -> None:
    payload = build_stroke_gait_analysis(
        {
            "avg_speed": 0.92,
            "avg_cadence": 106.0,
            "avg_asymmetry": 0.09,
            "avg_stability": 0.28,
            "avg_stride_time_cv": 3.0,
            "avg_step_time_asymmetry": 0.02,
            "tug_seconds": 21.0,
            "turn_duration_seconds": 4.2,
            "visual_gait_assessment": 3.0,
        }
    )
    assert payload["domain_scores"]["dynamic_stability"]["score"] == 2
    assert payload["indicators"]["prolonged_tug"]["abnormal"] is True
    assert payload["indicators"]["prolonged_turn_duration"]["abnormal"] is True


def test_build_stroke_gait_layers_separates_video_only_and_clinical_augmented() -> None:
    layers = build_stroke_gait_layers(
        {
            "avg_speed": 0.92,
            "avg_cadence": 106.0,
            "avg_asymmetry": 0.09,
            "avg_stability": 0.28,
            "avg_stride_time_cv": 3.0,
            "avg_step_time_asymmetry": 0.02,
            "tug_seconds": 21.0,
            "turn_duration_seconds": 4.2,
            "visual_gait_assessment": 3.0,
        }
    )
    assert layers["default_mode"] == "video_only"
    assert layers["video_only"]["mode"] == "video_only"
    assert layers["clinical_augmented"]["mode"] == "clinical_augmented"
    assert "prolonged_tug" not in layers["video_only"]["flagged_indicators"]
    assert any(
        indicator["label"] == "Prolonged TUG"
        for indicator in layers["clinical_augmented"]["flagged_indicators"]
    )
