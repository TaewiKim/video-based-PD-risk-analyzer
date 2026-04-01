import json

from django.db.utils import OperationalError, ProgrammingError
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .access_control import allowed_results, allowed_videos, can_access_result, remember_result_access
from .activity_schema import normalize_activity_schema
from .auth_utils import api_login_required
from .api_common import client_key, usage_snapshot
from .models import AnalysisResult
from .runtime import ALLOWED_EXTENSIONS, RESULTS_DIR, UPLOAD_DIR


@api_login_required
@require_GET
def api_results(request):
    visible_videos = allowed_videos(request)
    visible_results = allowed_results(request)
    try:
        items = [
            {
                "filename": row.result_filename,
                "type": row.result_type,
                "video_filename": row.video_filename or None,
                "size_bytes": len(json.dumps(row.payload, ensure_ascii=False).encode("utf-8")),
                "modified_ts": row.updated_at.timestamp(),
            }
            for row in AnalysisResult.objects.all().order_by("-updated_at")
            if (row.video_filename and row.video_filename in visible_videos)
            or row.result_filename in visible_results
        ]
        return JsonResponse(items, safe=False)
    except (OperationalError, ProgrammingError):
        pass

    items = []
    for path in sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        stem = path.stem
        video_filename = None
        for ext in ALLOWED_EXTENSIONS:
            candidate = UPLOAD_DIR / f"{stem.replace('_results', '').replace('_symptoms', '')}.{ext}"
            if candidate.exists():
                video_filename = candidate.name
                break
        items.append(
            {
                "filename": path.name,
                "type": "symptoms" if stem.endswith("_symptoms") else "gait",
                "video_filename": video_filename,
                "size_bytes": path.stat().st_size,
                "modified_ts": path.stat().st_mtime,
            }
        )
    filtered = [
        item
        for item in items
        if (item.get("video_filename") and item["video_filename"] in visible_videos)
        or item["filename"] in visible_results
    ]
    return JsonResponse(filtered, safe=False)


@api_login_required
@require_GET
def api_result_file(request, filename: str):
    try:
        row = AnalysisResult.objects.filter(result_filename=filename).first()
        if row is not None:
            if not (
                can_access_result(request, row.result_filename)
                or (row.video_filename and row.video_filename in allowed_videos(request))
            ):
                return JsonResponse({"error": "Forbidden"}, status=403)
            remember_result_access(request, row.result_filename)
            payload = normalize_activity_schema(row.payload)
            return JsonResponse(
                {
                    "result_filename": row.result_filename,
                    "video_filename": row.video_filename or None,
                    "data": payload,
                }
            )
    except (OperationalError, ProgrammingError):
        pass

    result_path = RESULTS_DIR / filename
    if not result_path.exists() or result_path.suffix.lower() != ".json":
        return JsonResponse({"error": "Result file not found"}, status=404)

    with open(result_path, encoding="utf-8") as f:
        payload = json.load(f)
    payload = normalize_activity_schema(payload)

    stem = result_path.stem
    video_filename = None
    for ext in ALLOWED_EXTENSIONS:
        candidate = UPLOAD_DIR / f"{stem.replace('_results', '').replace('_symptoms', '')}.{ext}"
        if candidate.exists():
            video_filename = candidate.name
            break
    if not (can_access_result(request, filename) or (video_filename and video_filename in allowed_videos(request))):
        return JsonResponse({"error": "Forbidden"}, status=403)
    remember_result_access(request, filename)

    return JsonResponse(
        {
            "result_filename": result_path.name,
            "video_filename": video_filename,
            "data": payload,
        }
    )


@api_login_required
@require_GET
def api_status(request):
    return JsonResponse(usage_snapshot(client_key(request)))
