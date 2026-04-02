from __future__ import annotations

from typing import Any

DEFAULT_STROKE_GAIT_THRESHOLDS: dict[str, Any] = {
    "speed": {"moderate": 0.8, "high": 0.4},
    "cadence": {"moderate": 100.0, "high": 90.0},
    "asymmetry": {"moderate": 0.2, "high": 0.35},
    "stability": {"moderate": 0.2, "high": 0.08},
    "tug_seconds": {"moderate": 13.5, "high": 20.0},
    "turn_duration_seconds": {"moderate": 2.8, "high": 4.0},
    "visual_gait_assessment": {"moderate": 1.5, "high": 2.5},
    "stride_time_cv": {"moderate": 4.0, "high": 8.0},
    "step_time_asymmetry": {"moderate": 0.05, "high": 0.08},
    "pattern_level": {"moderate_min": 2, "high_min": 5},
}

VOISARD_HS_CALIBRATED_THRESHOLDS: dict[str, Any] = {
    "speed": {"moderate": 0.72, "high": 0.35},
    "cadence": {"moderate": 96.0, "high": 84.0},
    "asymmetry": {"moderate": 0.22, "high": 0.38},
    "stability": {"moderate": 0.16, "high": 0.06},
    "tug_seconds": {"moderate": 15.0, "high": 22.0},
    "turn_duration_seconds": {"moderate": 3.0, "high": 4.4},
    "visual_gait_assessment": {"moderate": 2.0, "high": 3.0},
    "stride_time_cv": {"moderate": 5.5, "high": 9.5},
    "step_time_asymmetry": {"moderate": 0.06, "high": 0.09},
    "pattern_level": {"moderate_min": 2, "high_min": 6},
}

VIDEO_ONLY_THRESHOLD_PROFILES: dict[str, dict[str, Any]] = {
    "default": DEFAULT_STROKE_GAIT_THRESHOLDS,
    "voisard_hs_calibrated": VOISARD_HS_CALIBRATED_THRESHOLDS,
}


CLINICAL_AUGMENTED_THRESHOLD_PROFILES: dict[str, dict[str, Any]] = {
    "default": DEFAULT_STROKE_GAIT_THRESHOLDS,
    "voisard_hs_calibrated": VOISARD_HS_CALIBRATED_THRESHOLDS,
}


def _score_threshold(value: float, moderate_threshold: float, high_threshold: float, direction: str) -> int:
    if direction == "lt":
        if value < high_threshold:
            return 2
        if value < moderate_threshold:
            return 1
        return 0
    if value > high_threshold:
        return 2
    if value > moderate_threshold:
        return 1
    return 0


def _status_for_score(score: int) -> str:
    return "high" if score >= 2 else "moderate" if score == 1 else "low"


def _build_domain_result(
    *,
    name: str,
    score: int,
    evidence: list[dict[str, Any]],
    summary: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "score": score,
        "max_score": 2,
        "status": _status_for_score(score),
        "summary": summary,
        "evidence": evidence,
    }


def _resolve_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    merged = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in DEFAULT_STROKE_GAIT_THRESHOLDS.items()
    }
    for key, value in (thresholds or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def build_stroke_gait_analysis(
    summary: dict[str, Any] | None,
    *,
    thresholds: dict[str, Any] | None = None,
    include_clinical_metadata: bool = True,
) -> dict[str, Any]:
    config = _resolve_thresholds(thresholds)
    clinical_enabled = include_clinical_metadata
    if not summary:
        return {
            "available": False,
            "schema_version": "stroke_gait_analysis_v1",
            "analysis_type": "literature_rule_screen",
            "label": "Unavailable",
            "risk_level": "Unknown",
            "pattern_level": "Unknown",
            "risk_score": 0.0,
            "pattern_score": 0,
            "pattern_score_max": 8,
            "flagged_count": 0,
            "indicators": {},
            "domain_scores": {},
            "flagged_indicators": [],
            "summary_note": "No gait summary available.",
            "clinical_note": "Gait-derived screen unavailable.",
            "technical_note": "Stroke-linked gait analysis requires a usable gait summary.",
        }

    speed = float(summary.get("avg_speed", 0.0) or 0.0)
    asymmetry = float(summary.get("avg_asymmetry", 0.0) or 0.0)
    stability = float(summary.get("avg_stability", 0.0) or 0.0)
    cadence = float(summary.get("avg_cadence", 0.0) or 0.0)
    stride_time_cv = float(summary.get("avg_stride_time_cv", 0.0) or 0.0)
    step_time_asymmetry = float(summary.get("avg_step_time_asymmetry", 0.0) or 0.0)
    tug_seconds = float(summary.get("tug_seconds", 0.0) or 0.0)
    turn_duration_seconds = float(summary.get("turn_duration_seconds", 0.0) or 0.0)
    visual_gait_assessment = float(summary.get("visual_gait_assessment", 0.0) or 0.0)

    indicators: dict[str, dict[str, Any]] = {
        "gait_asymmetry": {
            "label": "Marked gait asymmetry",
            "value": round(asymmetry, 3),
            "moderate_threshold": config["asymmetry"]["moderate"],
            "high_threshold": config["asymmetry"]["high"],
            "direction": "gt",
            "score": _score_threshold(
                asymmetry,
                config["asymmetry"]["moderate"],
                config["asymmetry"]["high"],
                "gt",
            ),
            "unit": "ratio",
        },
        "low_stability": {
            "label": "Low gait stability",
            "value": round(stability, 3),
            "moderate_threshold": config["stability"]["moderate"],
            "high_threshold": config["stability"]["high"],
            "direction": "lt",
            "score": _score_threshold(
                stability,
                config["stability"]["moderate"],
                config["stability"]["high"],
                "lt",
            ),
            "unit": "score",
        },
        "low_cadence": {
            "label": "Reduced cadence",
            "value": round(cadence, 1),
            "moderate_threshold": config["cadence"]["moderate"],
            "high_threshold": config["cadence"]["high"],
            "direction": "lt",
            "score": _score_threshold(
                cadence,
                config["cadence"]["moderate"],
                config["cadence"]["high"],
                "lt",
            ),
            "unit": "steps/min",
        },
        "stride_variability": {
            "label": "High stride-time variability",
            "value": round(stride_time_cv, 2),
            "moderate_threshold": config["stride_time_cv"]["moderate"],
            "high_threshold": config["stride_time_cv"]["high"],
            "direction": "gt",
            "score": _score_threshold(
                stride_time_cv,
                config["stride_time_cv"]["moderate"],
                config["stride_time_cv"]["high"],
                "gt",
            ),
            "unit": "%",
        },
        "step_timing_asymmetry": {
            "label": "Step-time asymmetry",
            "value": round(step_time_asymmetry, 3),
            "moderate_threshold": config["step_time_asymmetry"]["moderate"],
            "high_threshold": config["step_time_asymmetry"]["high"],
            "direction": "gt",
            "score": _score_threshold(
                step_time_asymmetry,
                config["step_time_asymmetry"]["moderate"],
                config["step_time_asymmetry"]["high"],
                "gt",
            ),
            "unit": "ratio",
        },
    }
    if clinical_enabled:
        indicators["prolonged_tug"] = {
            "label": "Prolonged TUG",
            "value": round(tug_seconds, 2),
            "moderate_threshold": config["tug_seconds"]["moderate"],
            "high_threshold": config["tug_seconds"]["high"],
            "direction": "gt",
            "score": _score_threshold(
                tug_seconds,
                config["tug_seconds"]["moderate"],
                config["tug_seconds"]["high"],
                "gt",
            ),
            "unit": "s",
        }
        indicators["prolonged_turn_duration"] = {
            "label": "Prolonged turn duration",
            "value": round(turn_duration_seconds, 2),
            "moderate_threshold": config["turn_duration_seconds"]["moderate"],
            "high_threshold": config["turn_duration_seconds"]["high"],
            "direction": "gt",
            "score": _score_threshold(
                turn_duration_seconds,
                config["turn_duration_seconds"]["moderate"],
                config["turn_duration_seconds"]["high"],
                "gt",
            ),
            "unit": "s",
        }
        indicators["visual_gait_burden"] = {
            "label": "Elevated visual gait assessment",
            "value": round(visual_gait_assessment, 2),
            "moderate_threshold": config["visual_gait_assessment"]["moderate"],
            "high_threshold": config["visual_gait_assessment"]["high"],
            "direction": "gt",
            "score": _score_threshold(
                visual_gait_assessment,
                config["visual_gait_assessment"]["moderate"],
                config["visual_gait_assessment"]["high"],
                "gt",
            ),
            "unit": "score",
        }
    indicators["slow_speed"] = {
        "label": "Slow walking speed",
        "value": round(speed, 3),
        "moderate_threshold": config["speed"]["moderate"],
        "high_threshold": config["speed"]["high"],
        "direction": "lt",
        "score": _score_threshold(speed, config["speed"]["moderate"], config["speed"]["high"], "lt"),
        "unit": "m/s",
    }

    for item in indicators.values():
        item["abnormal"] = item["score"] > 0
        item["severity"] = _status_for_score(int(item["score"]))

    walking_capacity_score = max(indicators["slow_speed"]["score"], indicators["low_cadence"]["score"])
    spatial_asymmetry_score = indicators["gait_asymmetry"]["score"]
    temporal_variability_score = max(indicators["stride_variability"]["score"], indicators["step_timing_asymmetry"]["score"])
    dynamic_stability_inputs = [indicators["low_stability"]["score"]]
    if clinical_enabled:
        dynamic_stability_inputs.extend(
            [
                indicators["prolonged_tug"]["score"],
                indicators["prolonged_turn_duration"]["score"],
                indicators["visual_gait_burden"]["score"],
            ]
        )
    dynamic_stability_score = max(dynamic_stability_inputs)

    speed_band = (
        "household_only"
        if speed < config["speed"]["high"]
        else "limited_community"
        if speed < config["speed"]["moderate"]
        else "community_plus"
    )

    domain_scores = {
        "walking_capacity": _build_domain_result(
            name="Walking capacity",
            score=walking_capacity_score,
            evidence=[indicators["slow_speed"], indicators["low_cadence"]],
            summary=(
                "Walking speed and cadence are both reduced."
                if walking_capacity_score >= 2
                else "Walking capacity is below typical community ambulation range."
                if walking_capacity_score == 1
                else "Walking capacity remains within the expected community range."
            ),
        ),
        "spatial_asymmetry": _build_domain_result(
            name="Spatial asymmetry",
            score=spatial_asymmetry_score,
            evidence=[indicators["gait_asymmetry"]],
            summary=(
                "Marked left-right asymmetry is present."
                if spatial_asymmetry_score >= 2
                else "Mild to moderate asymmetry is present."
                if spatial_asymmetry_score == 1
                else "No material spatial asymmetry signal is flagged."
            ),
        ),
        "temporal_variability": _build_domain_result(
            name="Temporal variability",
            score=temporal_variability_score,
            evidence=[indicators["stride_variability"], indicators["step_timing_asymmetry"]],
            summary=(
                "Stride timing variability or step-timing asymmetry is markedly elevated."
                if temporal_variability_score >= 2
                else "Temporal gait consistency is mildly reduced."
                if temporal_variability_score == 1
                else "Temporal gait consistency is not materially flagged."
            ),
        ),
        "dynamic_stability": _build_domain_result(
            name="Dynamic stability",
            score=dynamic_stability_score,
            evidence=(
                [
                    indicators["low_stability"],
                    indicators["prolonged_tug"],
                    indicators["prolonged_turn_duration"],
                    indicators["visual_gait_burden"],
                ]
                if clinical_enabled
                else [indicators["low_stability"]]
            ),
            summary=(
                "Dynamic stability and functional mobility markers are substantially reduced."
                if dynamic_stability_score >= 2
                else "Dynamic stability or functional mobility shows a borderline reduction."
                if dynamic_stability_score == 1
                else "Dynamic stability is not materially flagged."
            ),
        ),
    }

    pattern_score = sum(domain["score"] for domain in domain_scores.values())
    pattern_level = (
        "High"
        if pattern_score >= config["pattern_level"]["high_min"]
        else "Moderate"
        if pattern_score >= config["pattern_level"]["moderate_min"]
        else "Low"
    )
    flagged_indicators = [item for item in indicators.values() if item["abnormal"]]
    flagged_count = len(flagged_indicators)
    risk_score = round(pattern_score / 8.0, 3)
    label = (
        "Post-stroke gait pattern is prominent"
        if pattern_level == "High"
        else "Post-stroke gait pattern is possible"
        if pattern_level == "Moderate"
        else "No dominant post-stroke gait pattern"
    )
    summary_note = (
        ", ".join(item["label"] for item in flagged_indicators)
        if flagged_indicators
        else "No major stroke-linked gait features are flagged."
    )

    return {
        "available": True,
        "schema_version": "stroke_gait_analysis_v1",
        "analysis_type": "literature_rule_screen",
        "label": label,
        "risk_level": pattern_level,
        "pattern_level": pattern_level,
        "risk_score": risk_score,
        "pattern_score": pattern_score,
        "pattern_score_max": 8,
        "flagged_count": flagged_count,
        "indicators": indicators,
        "domain_scores": domain_scores,
        "flagged_indicators": flagged_indicators,
        "speed_band": {
            "label": speed_band,
            "walking_speed_mps": round(speed, 3),
            "household_threshold_mps": config["speed"]["high"],
            "community_threshold_mps": config["speed"]["moderate"],
        },
        "summary_note": summary_note,
        "clinical_note": "Literature-based gait screen only. Do not use as a stroke diagnosis.",
        "technical_note": "Designed for gait-pattern triage and rehabilitation review, not cerebrovascular event prediction.",
        "threshold_profile": {
            "speed": config["speed"],
            "cadence": config["cadence"],
            "asymmetry": config["asymmetry"],
            "stability": config["stability"],
            "tug_seconds": config["tug_seconds"],
            "turn_duration_seconds": config["turn_duration_seconds"],
            "visual_gait_assessment": config["visual_gait_assessment"],
            "stride_time_cv": config["stride_time_cv"],
            "step_time_asymmetry": config["step_time_asymmetry"],
            "pattern_level": config["pattern_level"],
        },
    }


def build_stroke_gait_layers(
    summary: dict[str, Any] | None,
    *,
    video_only_profile: str = "default",
    clinical_augmented_profile: str = "voisard_hs_calibrated",
) -> dict[str, Any]:
    video_only_thresholds = VIDEO_ONLY_THRESHOLD_PROFILES.get(
        video_only_profile,
        DEFAULT_STROKE_GAIT_THRESHOLDS,
    )
    clinical_thresholds = CLINICAL_AUGMENTED_THRESHOLD_PROFILES.get(
        clinical_augmented_profile,
        VOISARD_HS_CALIBRATED_THRESHOLDS,
    )
    video_only = build_stroke_gait_analysis(
        summary,
        thresholds=video_only_thresholds,
        include_clinical_metadata=False,
    )
    clinical_augmented = build_stroke_gait_analysis(
        summary,
        thresholds=clinical_thresholds,
        include_clinical_metadata=True,
    )
    return {
        "default_mode": "video_only",
        "video_only": {
            **video_only,
            "mode": "video_only",
            "mode_label": "Video-only Stroke Gait Screen",
        },
        "clinical_augmented": {
            **clinical_augmented,
            "mode": "clinical_augmented",
            "mode_label": "Clinical-augmented Stroke Gait Screen",
        },
    }
