from typing import Any


ACTIVITY_SCHEMA_VERSION = "2026-03-25"


def _segment_duration(seg: dict[str, Any]) -> float:
    if seg.get("duration") is not None:
        return float(seg.get("duration", 0.0))
    start = float(seg.get("start_time", 0.0))
    end = float(seg.get("end_time", start))
    return max(0.0, end - start)


def build_routing_activity_schema(
    *,
    segments: list[dict[str, Any]] | None,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = summary or {}
    segments = segments or []
    total_duration = float(
        summary.get("total_duration")
        if summary.get("total_duration") is not None
        else (
            sum(_segment_duration(seg) for seg in segments)
            if segments
            else (
                float(summary.get("walking", 0.0))
                + float(summary.get("resting", 0.0))
                + float(summary.get("task", 0.0))
                + float(summary.get("standing", 0.0))
            )
        )
    )
    return {
        "kind": "symptom_routing",
        "labels": ["walking", "resting", "task", "standing"],
        "summary": {
            "walking": float(summary.get("walking", 0.0)),
            "resting": float(summary.get("resting", 0.0)),
            "task": float(summary.get("task", 0.0)),
            "standing": float(summary.get("standing", 0.0)),
            "total_duration": total_duration,
        },
        "segments": segments,
    }


def build_gait_phase_activity_schema(activity_timeline: dict[str, Any] | None) -> dict[str, Any] | None:
    activity_timeline = activity_timeline or {}
    summary = activity_timeline.get("summary")
    segments = activity_timeline.get("segments")
    if summary is None and not segments:
        return None
    summary = summary or {}
    total_duration = float(
        summary.get("total_duration")
        if summary.get("total_duration") is not None
        else (
            float(summary.get("walking_time", 0.0))
            + float(summary.get("turning_time", 0.0))
            + float(summary.get("standing_time", 0.0))
        )
    )
    return {
        "kind": "gait_phase",
        "labels": ["walking", "turning", "standing"],
        "summary": {
            "walking": float(summary.get("walking_time", 0.0)),
            "turning": float(summary.get("turning_time", 0.0)),
            "standing": float(summary.get("standing_time", 0.0)),
            "walking_ratio": float(summary.get("walking_ratio", 0.0)),
            "turning_ratio": float(summary.get("turning_ratio", 0.0)),
            "standing_ratio": float(summary.get("standing_ratio", 0.0)),
            "total_duration": total_duration,
        },
        "segments": segments or [],
    }


def build_activity_schema(
    *,
    routing_segments: list[dict[str, Any]] | None,
    routing_summary: dict[str, Any] | None,
    gait_timeline: dict[str, Any] | None = None,
    gait_source_person_id: str | None = None,
    owner_person_id: str | None = None,
) -> dict[str, Any]:
    gait_phase = build_gait_phase_activity_schema(gait_timeline)
    return {
        "version": ACTIVITY_SCHEMA_VERSION,
        "routing": build_routing_activity_schema(
            segments=routing_segments,
            summary=routing_summary,
        ),
        "gait_phase": (
            gait_phase
            if owner_person_id is None or gait_source_person_id is None or gait_source_person_id == owner_person_id
            else None
        ),
        "gait_source_person_id": gait_source_person_id,
        "owner_person_id": owner_person_id,
    }


def normalize_activity_schema(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    gait_analysis = payload.get("gait_analysis") or {}
    gait_source_person_id = gait_analysis.get("source_person_id")
    gait_timeline = gait_analysis.get("activity_timeline")

    if "activity_schema" not in payload:
        payload["activity_schema"] = build_activity_schema(
            routing_segments=[],
            routing_summary=payload.get("activity_summary") or {},
            gait_timeline=gait_timeline,
            gait_source_person_id=gait_source_person_id,
            owner_person_id=None,
        )
    else:
        root_schema = payload.get("activity_schema") or {}
        root_schema.setdefault("version", ACTIVITY_SCHEMA_VERSION)
        if root_schema.get("routing") is None:
            root_schema["routing"] = build_routing_activity_schema(
                segments=[],
                summary=payload.get("activity_summary") or {},
            )
        if root_schema.get("gait_phase") is None and gait_timeline:
            root_schema["gait_phase"] = build_gait_phase_activity_schema(gait_timeline)
        if root_schema.get("gait_source_person_id") is None and gait_source_person_id is not None:
            root_schema["gait_source_person_id"] = gait_source_person_id
        if "owner_person_id" not in root_schema:
            root_schema["owner_person_id"] = None
        payload["activity_schema"] = root_schema

    persons = payload.get("persons")
    if isinstance(persons, list):
        inferred_source = gait_source_person_id
        if inferred_source is None and len(persons) == 1:
            inferred_source = persons[0].get("person_id")
            if isinstance(gait_analysis, dict) and inferred_source:
                gait_analysis["source_person_id"] = inferred_source
        for person in persons:
            if not isinstance(person, dict):
                continue
            if "activity_schema" not in person:
                person["activity_schema"] = build_activity_schema(
                    routing_segments=person.get("activity_segments") or [],
                    routing_summary=person.get("activity_breakdown") or {},
                    gait_timeline=gait_timeline,
                    gait_source_person_id=inferred_source,
                    owner_person_id=person.get("person_id"),
                )
            else:
                person_schema = person.get("activity_schema") or {}
                person_schema.setdefault("version", ACTIVITY_SCHEMA_VERSION)
                if person_schema.get("routing") is None:
                    person_schema["routing"] = build_routing_activity_schema(
                        segments=person.get("activity_segments") or [],
                        summary=person.get("activity_breakdown") or {},
                    )
                if (
                    person_schema.get("gait_phase") is None
                    and gait_timeline
                    and inferred_source == person.get("person_id")
                ):
                    person_schema["gait_phase"] = build_gait_phase_activity_schema(gait_timeline)
                if person_schema.get("gait_source_person_id") is None and inferred_source is not None:
                    person_schema["gait_source_person_id"] = inferred_source
                if person_schema.get("owner_person_id") is None:
                    person_schema["owner_person_id"] = person.get("person_id")
                person["activity_schema"] = person_schema

            skeleton_track = person.get("skeleton_track")
            if isinstance(skeleton_track, dict):
                if skeleton_track.get("frames") is None and skeleton_track.get("keypoints") is not None:
                    skeleton_track["frames"] = skeleton_track.get("keypoints")
                if skeleton_track.get("keypoints") is None and skeleton_track.get("frames") is not None:
                    skeleton_track["keypoints"] = skeleton_track.get("frames")

    return payload
