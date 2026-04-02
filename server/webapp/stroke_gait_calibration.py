from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any

from .stroke_gait_analysis import build_stroke_gait_analysis


def extract_gait_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("summary") or payload.get("gait_analysis", {}).get("summary") or {}


def build_calibration_record_from_summary(
    path: str | Path,
    summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis = build_stroke_gait_analysis(summary)
    record = {
        "file": str(path),
        "summary": summary,
        "stroke_gait_analysis": analysis,
        "pattern_level": analysis.get("pattern_level", "Unknown"),
        "pattern_score": analysis.get("pattern_score", 0),
        "speed_band": analysis.get("speed_band", {}).get("label"),
        "flagged_count": analysis.get("flagged_count", 0),
    }
    if metadata:
        record["metadata"] = metadata
    return record


def build_calibration_record(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = extract_gait_summary(payload)
    return build_calibration_record_from_summary(path, summary)


def load_result_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def filter_calibration_records(
    records: list[dict[str, Any]],
    *,
    cohort_in: set[str] | None = None,
    cohort_exclude: set[str] | None = None,
    label_in: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata") or {}
        cohort = str(metadata.get("cohort") or "")
        label = str(metadata.get("label") or "")
        if cohort_in and cohort not in cohort_in:
            continue
        if cohort_exclude and cohort in cohort_exclude:
            continue
        if label_in and label not in label_in:
            continue
        filtered.append(record)
    return filtered


def build_group_comparison(
    records: list[dict[str, Any]],
    *,
    group_field: str = "cohort",
    groups: tuple[str, str] | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        metadata = record.get("metadata") or {}
        key = str(metadata.get(group_field) or "unknown")
        grouped.setdefault(key, []).append(record)

    if groups is None:
        if len(grouped) != 2:
            return {}
        groups = tuple(grouped.keys())  # type: ignore[assignment]

    left, right = groups
    left_records = grouped.get(left, [])
    right_records = grouped.get(right, [])
    if not left_records or not right_records:
        return {}

    def _mean(items: list[dict[str, Any]], key: str) -> float:
        values = [float(item.get(key, 0) or 0) for item in items]
        return round(sum(values) / len(values), 3) if values else 0.0

    def _share(items: list[dict[str, Any]], level: str) -> float:
        if not items:
            return 0.0
        count = sum(1 for item in items if item.get("pattern_level") == level)
        return round(count / len(items), 3)

    return {
        "group_field": group_field,
        "groups": {
            left: {
                "n_records": len(left_records),
                "mean_pattern_score": _mean(left_records, "pattern_score"),
                "high_share": _share(left_records, "High"),
                "moderate_or_higher_share": round(
                    sum(1 for item in left_records if item.get("pattern_level") in {"Moderate", "High"}) / len(left_records),
                    3,
                ),
            },
            right: {
                "n_records": len(right_records),
                "mean_pattern_score": _mean(right_records, "pattern_score"),
                "high_share": _share(right_records, "High"),
                "moderate_or_higher_share": round(
                    sum(1 for item in right_records if item.get("pattern_level") in {"Moderate", "High"}) / len(right_records),
                    3,
                ),
            },
        },
        "deltas": {
            "mean_pattern_score": round(_mean(left_records, "pattern_score") - _mean(right_records, "pattern_score"), 3),
            "high_share": round(_share(left_records, "High") - _share(right_records, "High"), 3),
        },
    }


def build_calibration_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts = Counter(record["pattern_level"] for record in records)
    speed_band_counts = Counter(record.get("speed_band") or "unknown" for record in records)
    scores = [int(record.get("pattern_score", 0) or 0) for record in records]
    flagged = [int(record.get("flagged_count", 0) or 0) for record in records]

    mean_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    mean_flagged = round(sum(flagged) / len(flagged), 3) if flagged else 0.0

    domain_counter: Counter[str] = Counter()
    cohort_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    for record in records:
        for key, domain in record.get("stroke_gait_analysis", {}).get("domain_scores", {}).items():
            if (domain or {}).get("score", 0) > 0:
                domain_counter[key] += 1
        metadata = record.get("metadata") or {}
        cohort = metadata.get("cohort")
        label = metadata.get("label")
        if cohort:
            cohort_counter[str(cohort)] += 1
        if label:
            label_counter[str(label)] += 1

    threshold_review = {
        "prominent_pattern_share": round(level_counts.get("High", 0) / len(records), 3) if records else 0.0,
        "limited_community_share": round(
            (speed_band_counts.get("household_only", 0) + speed_band_counts.get("limited_community", 0)) / len(records),
            3,
        ) if records else 0.0,
        "max_pattern_score": max(scores) if scores else 0,
        "min_pattern_score": min(scores) if scores else 0,
    }

    return {
        "n_records": len(records),
        "pattern_level_counts": dict(level_counts),
        "speed_band_counts": dict(speed_band_counts),
        "mean_pattern_score": mean_score,
        "mean_flagged_count": mean_flagged,
        "domain_flag_counts": dict(domain_counter),
        "cohort_counts": dict(cohort_counter),
        "label_counts": dict(label_counter),
        "threshold_review": threshold_review,
        "top_examples": sorted(
            [
                {
                    "file": record["file"],
                    "pattern_level": record["pattern_level"],
                    "pattern_score": record["pattern_score"],
                    "speed_band": record.get("speed_band"),
                    "metadata": record.get("metadata", {}),
                    "summary_note": record.get("stroke_gait_analysis", {}).get("summary_note"),
                }
                for record in records
            ],
            key=lambda item: (-int(item["pattern_score"]), item["file"]),
        )[:10],
    }
