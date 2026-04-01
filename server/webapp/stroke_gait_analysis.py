from __future__ import annotations

from typing import Any


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


def build_stroke_gait_analysis(summary: dict[str, Any] | None) -> dict[str, Any]:
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

    indicators: dict[str, dict[str, Any]] = {
        "slow_speed": {
            "label": "Slow walking speed",
            "value": round(speed, 3),
            "moderate_threshold": 0.8,
            "high_threshold": 0.4,
            "direction": "lt",
            "score": _score_threshold(speed, 0.8, 0.4, "lt"),
            "unit": "m/s",
        },
        "gait_asymmetry": {
            "label": "Marked gait asymmetry",
            "value": round(asymmetry, 3),
            "moderate_threshold": 0.2,
            "high_threshold": 0.35,
            "direction": "gt",
            "score": _score_threshold(asymmetry, 0.2, 0.35, "gt"),
            "unit": "ratio",
        },
        "low_stability": {
            "label": "Low gait stability",
            "value": round(stability, 3),
            "moderate_threshold": 0.2,
            "high_threshold": 0.08,
            "direction": "lt",
            "score": _score_threshold(stability, 0.2, 0.08, "lt"),
            "unit": "score",
        },
        "low_cadence": {
            "label": "Reduced cadence",
            "value": round(cadence, 1),
            "moderate_threshold": 100.0,
            "high_threshold": 90.0,
            "direction": "lt",
            "score": _score_threshold(cadence, 100.0, 90.0, "lt"),
            "unit": "steps/min",
        },
        "stride_variability": {
            "label": "High stride-time variability",
            "value": round(stride_time_cv, 2),
            "moderate_threshold": 4.0,
            "high_threshold": 8.0,
            "direction": "gt",
            "score": _score_threshold(stride_time_cv, 4.0, 8.0, "gt"),
            "unit": "%",
        },
        "step_timing_asymmetry": {
            "label": "Step-time asymmetry",
            "value": round(step_time_asymmetry, 3),
            "moderate_threshold": 0.05,
            "high_threshold": 0.08,
            "direction": "gt",
            "score": _score_threshold(step_time_asymmetry, 0.05, 0.08, "gt"),
            "unit": "ratio",
        },
    }

    for item in indicators.values():
        item["abnormal"] = item["score"] > 0
        item["severity"] = _status_for_score(int(item["score"]))

    walking_capacity_score = max(indicators["slow_speed"]["score"], indicators["low_cadence"]["score"])
    spatial_asymmetry_score = indicators["gait_asymmetry"]["score"]
    temporal_variability_score = max(indicators["stride_variability"]["score"], indicators["step_timing_asymmetry"]["score"])
    dynamic_stability_score = indicators["low_stability"]["score"]

    speed_band = (
        "household_only"
        if speed < 0.4
        else "limited_community"
        if speed < 0.8
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
            evidence=[indicators["low_stability"]],
            summary=(
                "Dynamic stability is substantially reduced."
                if dynamic_stability_score >= 2
                else "Dynamic stability shows a borderline reduction."
                if dynamic_stability_score == 1
                else "Dynamic stability is not materially flagged."
            ),
        ),
    }

    pattern_score = sum(domain["score"] for domain in domain_scores.values())
    pattern_level = "High" if pattern_score >= 5 else "Moderate" if pattern_score >= 2 else "Low"
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
            "household_threshold_mps": 0.4,
            "community_threshold_mps": 0.8,
        },
        "summary_note": summary_note,
        "clinical_note": "Literature-based gait screen only. Do not use as a stroke diagnosis.",
        "technical_note": "Designed for gait-pattern triage and rehabilitation review, not cerebrovascular event prediction.",
    }
