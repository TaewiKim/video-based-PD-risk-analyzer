import json

from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .activity_schema import normalize_activity_schema
from .api_common import client_key, usage_snapshot
from .runtime import ALLOWED_EXTENSIONS, RESULTS_DIR, UPLOAD_DIR


@require_GET
def api_results(request):
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
    return JsonResponse(items, safe=False)


@require_GET
def api_result_file(request, filename: str):
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

    return JsonResponse(
        {
            "result_filename": result_path.name,
            "video_filename": video_filename,
            "data": payload,
        }
    )


@require_GET
def api_status(request):
    return JsonResponse(usage_snapshot(client_key(request)))
