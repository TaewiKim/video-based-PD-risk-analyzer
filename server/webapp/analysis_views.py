import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .activity_schema import build_activity_schema, normalize_activity_schema
from .api_common import (
    client_key,
    consume_usage,
    derive_fog_transitions,
    ensure_analyzers,
    parse_json_body,
)
from .runtime import RESULTS_DIR, UPLOAD_DIR
from .models import AnalysisResult


def _persist_result_record(result_filename: str, result_type: str, video_filename: str, payload: dict) -> None:
    AnalysisResult.objects.update_or_create(
        result_filename=result_filename,
        defaults={
            "result_type": result_type,
            "video_filename": video_filename,
            "payload": payload,
        },
    )


@csrf_exempt
@require_POST
def api_analyze(request):
    payload = parse_json_body(request)
    filename = payload.get("filename")
    identify_user = bool(payload.get("identify_user", True))
    if not filename:
        return JsonResponse({"error": "No filename provided"}, status=400)

    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return JsonResponse({"error": "File not found"}, status=404)

    analyzer, _ = ensure_analyzers()
    _, usage = consume_usage(client_key(request))

    try:
        results = analyzer.analyze_video(str(filepath), identify_user=identify_user)
        result_filename = f"{filename.rsplit('.', 1)[0]}_results.json"
        with open(RESULTS_DIR / result_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        _persist_result_record(result_filename, "gait", filename, results)
        return JsonResponse({**results, "usage": usage})
    except Exception as exc:
        return JsonResponse({"error": str(exc), **usage}, status=500)


@csrf_exempt
@require_POST
def api_analyze_symptoms(request):
    payload = parse_json_body(request)
    filename = payload.get("filename")
    symptoms = payload.get("symptoms")
    if not filename:
        return JsonResponse({"error": "No filename provided"}, status=400)

    filepath = UPLOAD_DIR / filename
    if not filepath.exists():
        return JsonResponse({"error": "File not found"}, status=404)

    analyzer, pd_symptoms_analyzer = ensure_analyzers()
    _, usage = consume_usage(client_key(request))

    try:
        tracks, video_info = pd_symptoms_analyzer.tracker.extract_tracks(
            str(filepath),
            extract_hands=True,
            extract_face=True,
        )
        results = pd_symptoms_analyzer.analyze_tracks(
            tracks=tracks,
            video_info=video_info,
            symptoms=symptoms,
            include_skeleton=True,
            skeleton_frame_stride=1,
        )

        primary_track = max(tracks, key=lambda track: track.duration_frames, default=None)
        primary_person_id = primary_track.person_id if primary_track is not None else None

        gait_results = {}
        if primary_track is not None:
            gait_results = analyzer.analyze_precomputed_pose(
                raw_keypoints=primary_track.keypoints,
                raw_pose_quality=primary_track.confidence_scores
                if primary_track.confidence_scores is not None
                else None,
                fps=float(video_info.get("fps", 0.0)),
                total_frames=int(video_info.get("total_frames", primary_track.duration_frames)),
                width=int(video_info.get("width", 0)),
                height=int(video_info.get("height", 0)),
                video_path=str(filepath),
                pose_backend=str(video_info.get("pose_backend", "unknown")),
                time_offset_sec=primary_track.start_frame / max(float(video_info.get("fps", 1.0)), 1e-6),
                frame_offset=int(primary_track.start_frame),
                video_duration=float(video_info.get("duration", 0.0)),
            )

        walking_segments = gait_results.get("walking_detection", {}).get("segments", [])
        video_duration = float(gait_results.get("video_info", {}).get("duration", 60))
        fog_transitions = derive_fog_transitions(walking_segments, video_duration)

        summary = gait_results.get("summary", {}) or {}
        walking_detection = gait_results.get("walking_detection", {}) or {}
        turn_detection = gait_results.get("turn_detection", {}) or {}
        turn_analysis = gait_results.get("turn_analysis", []) or []
        activity_timeline = gait_results.get("activity_timeline", {}) or {}
        turn_methodology = gait_results.get("turn_methodology", {}) or {}
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
            "video_info": gait_results.get("video_info", {}),
            "user": gait_results.get("user"),
            "pose_backend": gait_results.get("pose_backend"),
            "gait_review_track": gait_results.get("gait_review_track"),
            "walking_detection": walking_detection,
            "analysis_results": analysis_results,
            "statistical_analysis": gait_results.get("statistical_analysis", {}),
            "preprocessing": gait_results.get("preprocessing", {}),
            "ml_inference": gait_results.get("ml_inference", {}),
            "summary": summary,
            "turn_detection": turn_detection,
            "turn_analysis": turn_analysis,
            "turn_methodology": turn_methodology,
            "activity_timeline": activity_timeline,
            "source_person_id": primary_person_id,
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

        results["activity_schema"] = build_activity_schema(
            routing_segments=[],
            routing_summary=results.get("activity_summary") or {},
            gait_timeline=activity_timeline,
            gait_source_person_id=primary_person_id,
            owner_person_id=None,
        )

        for person in results.get("persons", []):
            fog_result = person.get("symptoms", {}).get("fog")
            if fog_result is not None:
                fog_result["transitions_detected"] = len(fog_transitions)
                fog_result["transitions"] = fog_transitions
                fog_result["turn_detection"] = turn_detection
                fog_result["turn_analysis"] = turn_analysis
                fog_result["turn_methodology"] = turn_methodology
            person["activity_schema"] = build_activity_schema(
                routing_segments=person.get("activity_segments") or [],
                routing_summary=person.get("activity_breakdown") or {},
                gait_timeline=activity_timeline,
                gait_source_person_id=primary_person_id,
                owner_person_id=person.get("person_id"),
            )

        results = normalize_activity_schema(results)

        result_filename = f"{filename.rsplit('.', 1)[0]}_symptoms.json"
        with open(RESULTS_DIR / result_filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        _persist_result_record(result_filename, "symptoms", filename, results)

        return JsonResponse({**results, "usage": usage})
    except Exception as exc:
        return JsonResponse({"error": str(exc), **usage}, status=500)


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
