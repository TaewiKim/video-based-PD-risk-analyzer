import base64
import mimetypes
import uuid

import cv2
import numpy as np
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .api_common import ensure_analyzers, parse_json_body
from .runtime import ALLOWED_EXTENSIONS, PROFILE_DIR, UPLOAD_DIR


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@csrf_exempt
@require_POST
def api_upload(request):
    video = request.FILES.get("video")
    if video is None:
        return JsonResponse({"error": "No video file provided"}, status=400)
    if not video.name or not _allowed_file(video.name):
        return JsonResponse({"error": "Invalid file type"}, status=400)

    ext = video.name.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = UPLOAD_DIR / filename
    with open(filepath, "wb") as f:
        for chunk in video.chunks():
            f.write(chunk)

    return JsonResponse({"success": True, "filename": filename, "video_url": f"/videos/{filename}"})


@csrf_exempt
@require_POST
def api_register_user(request):
    payload = parse_json_body(request)
    name = payload.get("name")
    image_data = payload.get("image")
    if not name:
        return JsonResponse({"error": "Name is required"}, status=400)
    if not image_data:
        return JsonResponse({"error": "Face image is required"}, status=400)

    analyzer, _ = ensure_analyzers()
    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JsonResponse({"error": "Invalid image data"}, status=400)

        profile = analyzer.register_user(name, image)
        if profile:
            return JsonResponse({"success": True, "user": profile})
        return JsonResponse({"error": "Face not detected in image"}, status=400)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_GET
def api_users(request):
    analyzer, _ = ensure_analyzers()
    return JsonResponse(analyzer.list_users(), safe=False)


@require_GET
def user_photo(request, user_id: str):
    photo_path = PROFILE_DIR / f"{user_id}_face.jpg"
    if not photo_path.exists():
        return JsonResponse({"error": "Photo not found"}, status=404)
    return FileResponse(open(photo_path, "rb"), content_type="image/jpeg")


@require_GET
def video_file(request, filename: str):
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return JsonResponse({"error": "Video not found"}, status=404)
    content_type, _ = mimetypes.guess_type(str(video_path))
    return FileResponse(open(video_path, "rb"), content_type=content_type or "application/octet-stream")
