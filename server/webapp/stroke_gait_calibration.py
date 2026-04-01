from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any

from .stroke_gait_analysis import build_stroke_gait_analysis


def extract_gait_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("summary") or payload.get("gait_analysis", {}).get("summary") or {}


def build_calibration_record(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary = extract_gait_summary(payload)
    analysis = build_stroke_gait_analysis(summary)
    return {
        "file": str(path),
        "summary": summary,
        "stroke_gait_analysis": analysis,
        "pattern_level": analysis.get("pattern_level", "Unknown"),
        "pattern_score": analysis.get("pattern_score", 0),
        "speed_band": analysis.get("speed_band", {}).get("label"),
        "flagged_count": analysis.get("flagged_count", 0),
    }


def load_result_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_calibration_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts = Counter(record["pattern_level"] for record in records)
    speed_band_counts = Counter(record.get("speed_band") or "unknown" for record in records)
    scores = [int(record.get("pattern_score", 0) or 0) for record in records]
    flagged = [int(record.get("flagged_count", 0) or 0) for record in records]

    mean_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    mean_flagged = round(sum(flagged) / len(flagged), 3) if flagged else 0.0

    domain_counter: Counter[str] = Counter()
    for record in records:
        for key, domain in record.get("stroke_gait_analysis", {}).get("domain_scores", {}).items():
            if (domain or {}).get("score", 0) > 0:
                domain_counter[key] += 1

    return {
        "n_records": len(records),
        "pattern_level_counts": dict(level_counts),
        "speed_band_counts": dict(speed_band_counts),
        "mean_pattern_score": mean_score,
        "mean_flagged_count": mean_flagged,
        "domain_flag_counts": dict(domain_counter),
        "top_examples": sorted(
            [
                {
                    "file": record["file"],
                    "pattern_level": record["pattern_level"],
                    "pattern_score": record["pattern_score"],
                    "speed_band": record.get("speed_band"),
                    "summary_note": record.get("stroke_gait_analysis", {}).get("summary_note"),
                }
                for record in records
            ],
            key=lambda item: (-int(item["pattern_score"]), item["file"]),
        )[:10],
    }
