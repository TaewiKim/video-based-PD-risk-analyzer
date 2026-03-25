import json
from typing import Any

from django.utils import timezone as dj_timezone

from .activity_schema import normalize_activity_schema
from .models import PersonUsage as PersonUsageModel
from .models import UsageEvent as UsageEventModel
from .runtime import DATA_DIR
from .services.pd_symptoms_analyzer import get_pd_symptoms_analyzer
from .services.smart_analyzer import get_analyzer

PersonUsage: Any = PersonUsageModel
UsageEvent: Any = UsageEventModel

_analyzer = None
_pd_symptoms_analyzer = None


def ensure_analyzers() -> tuple[Any, Any]:
    global _analyzer, _pd_symptoms_analyzer
    if _analyzer is not None and _pd_symptoms_analyzer is not None:
        return _analyzer, _pd_symptoms_analyzer

    _analyzer = get_analyzer(str(DATA_DIR))
    _pd_symptoms_analyzer = get_pd_symptoms_analyzer()
    return _analyzer, _pd_symptoms_analyzer


def parse_json_body(request) -> dict[str, Any]:
    return json.loads(request.body.decode("utf-8")) if request.body else {}


def normalize_person_id(raw: str) -> str:
    return (raw or "").strip().lower()


def client_key(request) -> str:
    explicit = request.headers.get("X-Client-Id", "")
    if explicit.strip():
        return normalize_person_id(explicit)
    ip = request.META.get("REMOTE_ADDR", "anonymous")
    return f"ip:{ip}"


def usage_snapshot(raw_client_key: str) -> dict[str, Any]:
    person, _ = PersonUsage.objects.get_or_create(
        person_id=raw_client_key, defaults={"used_count": 0}
    )
    today = dj_timezone.localdate()
    used_today = UsageEvent.objects.filter(person=person, created_at__date=today).count()
    return {
        "client_id": person.person_id,
        "used_count": int(used_today),
        "remaining": None,
        "limit": None,
        "unlimited": True,
    }


def consume_usage(raw_client_key: str) -> tuple[bool, dict[str, Any]]:
    person, _ = PersonUsage.objects.get_or_create(
        person_id=raw_client_key, defaults={"used_count": 0}
    )
    person.used_count += 1
    person.save(update_fields=["used_count", "updated_at"])
    UsageEvent.objects.create(person=person)
    return True, usage_snapshot(raw_client_key)


def derive_fog_transitions(
    walking_segments: list[dict[str, Any]], video_duration: float
) -> list[dict[str, Any]]:
    if not walking_segments:
        return []

    transitions: list[dict[str, Any]] = []
    sorted_segments = sorted(walking_segments, key=lambda s: float(s.get("start_time", 0)))

    for idx, seg in enumerate(sorted_segments):
        start = float(seg.get("start_time", 0))
        end = float(seg.get("end_time", start))

        pre_standing_duration = (
            start if idx == 0 else start - float(sorted_segments[idx - 1].get("end_time", 0))
        )
        if pre_standing_duration >= 1.0:
            transitions.append(
                {
                    "type": "initiation",
                    "transition_type": "standing_to_walking",
                    "boundary_time": start,
                    "standing_duration": pre_standing_duration,
                    "walking_segment_idx": idx,
                }
            )

        post_standing_duration = (
            video_duration - end
            if idx == len(sorted_segments) - 1
            else float(sorted_segments[idx + 1].get("start_time", end)) - end
        )
        if post_standing_duration >= 1.0:
            transitions.append(
                {
                    "type": "termination",
                    "transition_type": "walking_to_standing",
                    "boundary_time": end,
                    "standing_duration": post_standing_duration,
                    "walking_segment_idx": idx,
                }
            )
    return transitions
