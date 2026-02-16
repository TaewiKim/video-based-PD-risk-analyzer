import base64
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.http import FileResponse, JsonResponse
from django.http import HttpResponse
from django.shortcuts import render
from django.utils import timezone as dj_timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .models import PersonUsage as PersonUsageModel
from .models import UsageEvent as UsageEventModel

PersonUsage: Any = PersonUsageModel
UsageEvent: Any = UsageEventModel

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
UPLOAD_DIR = WEB_DIR / "uploads"
RESULTS_DIR = WEB_DIR / "results"
DATA_DIR = WEB_DIR / "data"
PROFILE_DIR = DATA_DIR / "profiles"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "webm", "mkv"}

for folder in [UPLOAD_DIR, RESULTS_DIR, DATA_DIR, PROFILE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def _load_function(module_name: str, file_path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


_analyzer = None
_pd_symptoms_analyzer = None


def _ensure_analyzers() -> tuple[Any, Any]:
    global _analyzer, _pd_symptoms_analyzer
    if _analyzer is not None and _pd_symptoms_analyzer is not None:
        return _analyzer, _pd_symptoms_analyzer

    get_analyzer = _load_function(
        "web_smart_analyzer", WEB_DIR / "smart_analyzer.py", "get_analyzer"
    )
    get_pd_symptoms_analyzer = _load_function(
        "web_pd_symptoms_analyzer",
        WEB_DIR / "pd_symptoms_analyzer.py",
        "get_pd_symptoms_analyzer",
    )
    _analyzer = get_analyzer(str(DATA_DIR))
    _pd_symptoms_analyzer = get_pd_symptoms_analyzer()
    return _analyzer, _pd_symptoms_analyzer


def _normalize_person_id(raw: str) -> str:
    return (raw or "").strip().lower()


def _client_key(request) -> str:
    explicit = request.headers.get("X-Client-Id", "")
    if explicit.strip():
        return _normalize_person_id(explicit)
    ip = request.META.get("REMOTE_ADDR", "anonymous")
    return f"ip:{ip}"


def _usage_snapshot(client_key: str) -> dict[str, Any]:
    person, _ = PersonUsage.objects.get_or_create(person_id=client_key, defaults={"used_count": 0})
    today = dj_timezone.localdate()
    used_today = UsageEvent.objects.filter(person=person, created_at__date=today).count()
    return {
        "client_id": person.person_id,
        "used_count": int(used_today),
        "remaining": None,
        "limit": None,
        "unlimited": True,
    }


def _consume_usage(client_key: str) -> tuple[bool, dict[str, Any]]:
    person, _ = PersonUsage.objects.get_or_create(person_id=client_key, defaults={"used_count": 0})
    person.used_count += 1
    person.save(update_fields=["used_count", "updated_at"])
    UsageEvent.objects.create(person=person)
    return True, _usage_snapshot(client_key)


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _derive_fog_transitions(
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


def index(request):
    return render(request, "index.html")


@require_GET
def api_status(request):
    return JsonResponse(_usage_snapshot(_client_key(request)))


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
def api_analyze(request):
    payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    filename = payload.get("filename")
    identify_user = bool(payload.get("identify_user", True))
    if not filename:
        return JsonResponse({"error": "No filename provided"}, status=400)

    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return JsonResponse({"error": "File not found"}, status=404)

    analyzer, _ = _ensure_analyzers()
    _, usage = _consume_usage(_client_key(request))

    try:
        results = analyzer.analyze_video(str(filepath), identify_user=identify_user)
        with open(
            RESULTS_DIR / f"{filename.rsplit('.', 1)[0]}_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2, default=str)
        return JsonResponse({**results, "usage": usage})
    except Exception as exc:
        return JsonResponse({"error": str(exc), **usage}, status=500)


@csrf_exempt
@require_POST
def api_analyze_symptoms(request):
    payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    filename = payload.get("filename")
    symptoms = payload.get("symptoms")
    if not filename:
        return JsonResponse({"error": "No filename provided"}, status=400)

    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return JsonResponse({"error": "File not found"}, status=404)

    analyzer, pd_symptoms_analyzer = _ensure_analyzers()
    _, usage = _consume_usage(_client_key(request))

    try:
        results = pd_symptoms_analyzer.analyze_video(
            str(filepath), symptoms=symptoms, include_skeleton=True, skeleton_frame_stride=2
        )

        gait_results = analyzer.analyze_video(str(filepath), identify_user=False)
        walking_segments = gait_results.get("walking_detection", {}).get("segments", [])
        video_duration = float(gait_results.get("video_info", {}).get("duration", 60))
        fog_transitions = _derive_fog_transitions(walking_segments, video_duration)

        summary = gait_results.get("summary", {}) or {}
        walking_detection = gait_results.get("walking_detection", {}) or {}
        analysis_results = gait_results.get("analysis_results", []) or []

        estimated_steps = 0
        for seg in analysis_results:
            cadence = seg.get("cadence")
            duration = seg.get("duration")
            if cadence is None or duration is None:
                continue
            estimated_steps += int(round((float(cadence) * float(duration)) / 60.0))

        results["gait_analysis"] = {
            "success": True,
            "walking_detection": walking_detection,
            "analysis_results": analysis_results,
            "statistical_analysis": gait_results.get("statistical_analysis", {}),
            "preprocessing": gait_results.get("preprocessing", {}),
            "ml_inference": gait_results.get("ml_inference", {}),
            "summary": summary,
            "biomarkers": {
                "stride_cv": summary.get("avg_stride_time_cv"),
                "arm_swing_asymmetry": summary.get("avg_arm_swing_asymmetry"),
                "step_time_asymmetry": summary.get("avg_step_time_asymmetry"),
                "pd_risk_score": summary.get("avg_pd_risk_score"),
                "walk_ratio": walking_detection.get("walking_ratio"),
            },
            "gait_metrics": {
                "walking_speed": summary.get("avg_speed"),
                "stride_length": summary.get("avg_stride_length"),
                "cadence": summary.get("avg_cadence"),
                "step_count": estimated_steps,
            },
            "classification": summary.get("overall_classification", "Unknown"),
            "pd_risk_score": float(summary.get("avg_pd_risk_score", 0)) * 100,
            "segments": walking_segments,
            "fog_transitions": fog_transitions,
            "fog_transition_count": len(fog_transitions),
        }

        for person in results.get("persons", []):
            fog_result = person.get("symptoms", {}).get("fog")
            if fog_result:
                fog_result["transitions_detected"] = len(fog_transitions)
                fog_result["transitions"] = fog_transitions

        with open(
            RESULTS_DIR / f"{filename.rsplit('.', 1)[0]}_symptoms.json", "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2, default=str)

        return JsonResponse({**results, "usage": usage})
    except Exception as exc:
        return JsonResponse({"error": str(exc), **usage}, status=500)


@csrf_exempt
@require_POST
def api_register_user(request):
    payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    name = payload.get("name")
    image_data = payload.get("image")
    if not name:
        return JsonResponse({"error": "Name is required"}, status=400)
    if not image_data:
        return JsonResponse({"error": "Face image is required"}, status=400)

    analyzer, _ = _ensure_analyzers()
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
    analyzer, _ = _ensure_analyzers()
    return JsonResponse(analyzer.list_users(), safe=False)


@require_GET
def api_reference_data(request):
    return JsonResponse(
        {
            "healthy_young": {"mean": 1.24, "std": 0.18, "n": 24},
            "healthy_older": {"mean": 1.21, "std": 0.19, "n": 18},
            "pd_off": {"mean": 0.86, "std": 0.30, "n": 23},
            "pd_on": {"mean": 1.02, "std": 0.28, "n": 25},
            "model_info": {
                "classifier": "HistGradientBoosting",
                "dataset": "CARE-PD",
                "dataset_size": 2953,
                "binary_accuracy": 0.890,
                "binary_roc_auc": 0.957,
                "binary_threshold": 0.535,
                "binary_threshold_tuned_accuracy": 0.8947,
                "binary_threshold_selection": "OOF sweep 0.05-0.95 (max Accuracy, tie-break Macro-F1)",
                "multiclass_accuracy": 0.813,
                "multiclass_balanced_accuracy": 0.800,
                "multiclass_macro_f1": 0.814,
                "preprocessing": {
                    "target_fps": 30,
                    "alignment": "origin + PCA forward-axis",
                    "velocity_outlier_clip_percentile": 99,
                },
            },
        }
    )


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
    return FileResponse(open(video_path, "rb"), content_type="video/mp4")


@require_GET
def favicon(request):
    return HttpResponse(status=204)
