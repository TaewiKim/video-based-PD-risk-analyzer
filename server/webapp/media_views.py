import base64
import mimetypes
import subprocess
import uuid

import cv2
import numpy as np
from django.http import FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .access_control import (
    allowed_users,
    can_access_user,
    can_access_video,
    remember_user_access,
    remember_video_access,
)
from .auth_utils import api_login_required
from .api_common import ensure_analyzers, parse_json_body
from .runtime import ALLOWED_EXTENSIONS, PROFILE_DIR, UPLOAD_DIR

ANALYSIS_MAX_SIDE = 320


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _normalized_dimensions(video_path, max_side: int = ANALYSIS_MAX_SIDE):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        raise ValueError("Failed to read uploaded video dimensions")

    scale = min(1.0, max_side / float(max(width, height)))
    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))
    if target_width % 2:
        target_width -= 1
    if target_height % 2:
        target_height -= 1
    return max(2, target_width), max(2, target_height)


def _normalize_video_for_analysis(source_path, target_path):
    target_width, target_height = _normalized_dimensions(source_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        f"scale={target_width}:{target_height}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _ingest_processing_metadata(source_extension: str) -> dict:
    return {
        "normalized_for_analysis": True,
        "source_extension": source_extension,
        "output_extension": "mp4",
        "video_codec": "h264",
        "audio_removed": True,
        "max_side_px": ANALYSIS_MAX_SIDE,
        "notice": (
            "Uploaded videos are resized, transcoded to MP4/H.264, and audio is removed before analysis."
        ),
    }


@api_login_required
@csrf_exempt
@require_POST
def api_upload(request):
    video = request.FILES.get("video")
    if video is None:
        return JsonResponse({"error": "No video file provided"}, status=400)
    if not video.name or not _allowed_file(video.name):
        return JsonResponse({"error": "Invalid file type"}, status=400)

    upload_id = uuid.uuid4().hex
    ext = video.name.rsplit(".", 1)[1].lower()
    raw_path = UPLOAD_DIR / f"{upload_id}.source.{ext}"
    filename = f"{upload_id}.mp4"
    filepath = UPLOAD_DIR / filename
    with open(raw_path, "wb") as f:
        for chunk in video.chunks():
            f.write(chunk)
    try:
        _normalize_video_for_analysis(raw_path, filepath)
    except Exception as exc:
        if filepath.exists():
            filepath.unlink()
        raw_path.unlink(missing_ok=True)
        return JsonResponse({"error": f"Failed to normalize video: {exc}"}, status=500)
    raw_path.unlink(missing_ok=True)
    remember_video_access(request, filename)
    return JsonResponse(
        {
            "success": True,
            "filename": filename,
            "video_url": f"/videos/{filename}",
            "input_processing": _ingest_processing_metadata(ext),
        }
    )


@api_login_required
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
            remember_user_access(request, profile.get("user_id"))
            return JsonResponse({"success": True, "user": profile})
        return JsonResponse({"error": "Face not detected in image"}, status=400)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@api_login_required
@require_GET
def api_users(request):
    analyzer, _ = ensure_analyzers()
    visible_ids = allowed_users(request)
    users = [user for user in analyzer.list_users() if user.get("user_id") in visible_ids]
    return JsonResponse(users, safe=False)


@api_login_required
@require_GET
def user_photo(request, user_id: str):
    if not can_access_user(request, user_id):
        return JsonResponse({"error": "Forbidden"}, status=403)
    photo_path = PROFILE_DIR / f"{user_id}_face.jpg"
    if not photo_path.exists():
        return JsonResponse({"error": "Photo not found"}, status=404)
    return FileResponse(open(photo_path, "rb"), content_type="image/jpeg")


@api_login_required
@require_GET
def video_file(request, filename: str):
    if not can_access_video(request, filename):
        return JsonResponse({"error": "Forbidden"}, status=403)
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return JsonResponse({"error": "Video not found"}, status=404)
    content_type, _ = mimetypes.guess_type(str(video_path))
    return FileResponse(open(video_path, "rb"), content_type=content_type or "application/octet-stream")
