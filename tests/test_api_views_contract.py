"""Django view-level API contract tests.

These tests patch the heavy analyzers and persistence boundaries so the
response contract can be validated without running real CV inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.config.settings.test")

import django

django.setup()

from django.contrib.sessions.middleware import SessionMiddleware
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory

from server.webapp.access_control import remember_video_access
from server.webapp.analysis_views import (
    api_analyze,
    api_analyze_async,
    api_analyze_symptoms,
    api_analyze_symptoms_async,
    api_job_status,
)
from server.webapp.media_views import api_upload, api_users, user_photo, video_file
from server.webapp.results_views import api_result_file, api_results, api_status

pytestmark = pytest.mark.django_db


def _json(response):
    return json.loads(response.content.decode("utf-8"))


def _attach_session(request):
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.user = SimpleNamespace(
        pk=1,
        is_authenticated=True,
        is_staff=False,
        is_superuser=False,
        get_username=lambda: "tester",
    )
    request.session.save()
    return request


def test_api_status_returns_usage_snapshot() -> None:
    factory = RequestFactory()
    request = _attach_session(factory.get("/api/status"))

    with patch(
        "server.webapp.results_views.usage_snapshot",
        return_value={
            "client_id": "ip:test",
            "used_count": 2,
            "remaining": None,
            "limit": None,
            "unlimited": True,
        },
    ):
        response = api_status(request)

    payload = _json(response)
    assert response.status_code == 200
    assert payload["client_id"] == "ip:test"
    assert payload["used_count"] == 2


def test_api_upload_rejects_missing_video() -> None:
    factory = RequestFactory()
    request = _attach_session(factory.post("/api/upload", data={}))

    response = api_upload(request)

    payload = _json(response)
    assert response.status_code == 400
    assert payload["error"] == "No video file provided"


def test_api_upload_rejects_invalid_extension() -> None:
    factory = RequestFactory()
    upload = SimpleUploadedFile("bad.txt", b"not-video", content_type="text/plain")
    request = _attach_session(factory.post("/api/upload", data={"video": upload}))

    response = api_upload(request)

    payload = _json(response)
    assert response.status_code == 400
    assert payload["error"] == "Invalid file type"


def test_api_upload_normalizes_to_mp4_contract() -> None:
    factory = RequestFactory()

    with TemporaryDirectory() as tmpdir:
        upload_dir = Path(tmpdir)
        upload = SimpleUploadedFile("clip.webm", b"video-bytes", content_type="video/webm")
        request = _attach_session(factory.post("/api/upload", data={"video": upload}))

        def fake_normalize(source_path, target_path):
            Path(target_path).write_bytes(b"normalized-video")

        with (
            patch("server.webapp.media_views.UPLOAD_DIR", upload_dir),
            patch("server.webapp.media_views._normalize_video_for_analysis", side_effect=fake_normalize),
            patch("server.webapp.media_views.uuid.uuid4", return_value=SimpleNamespace(hex="abc123")),
        ):
            response = api_upload(request)

        payload = _json(response)
        assert response.status_code == 200
        assert payload["success"] is True
        assert payload["filename"] == "abc123.mp4"
        assert payload["video_url"] == "/videos/abc123.mp4"
        assert payload["input_processing"]["audio_removed"] is True
        assert request.session["allowed_video_filenames"] == ["abc123.mp4"]
        assert (upload_dir / "abc123.mp4").exists()
        assert not (upload_dir / "abc123.source.webm").exists()


def test_api_analyze_requires_session_video_access() -> None:
    factory = RequestFactory()
    request = _attach_session(
        factory.post(
            "/api/analyze",
            data=json.dumps({"filename": "sample.mp4", "identify_user": False}),
            content_type="application/json",
        )
    )

    response = api_analyze(request)
    payload = _json(response)
    assert response.status_code == 403
    assert payload["error"] == "Forbidden"


def test_api_analyze_returns_expected_contract() -> None:
    factory = RequestFactory()

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        upload_dir = tmp / "uploads"
        results_dir = tmp / "results"
        upload_dir.mkdir()
        results_dir.mkdir()
        (upload_dir / "sample.mp4").write_bytes(b"video")

        fake_results = {
            "video_info": {"duration": 5.0, "fps": 30.0},
            "user": None,
            "walking_detection": {"segments": [], "walking_ratio": 0.0},
            "preprocessing": {"detection_rate": 0.9, "mean_pose_quality": 0.8},
            "analysis_results": [],
            "summary": {"overall_classification": "Unknown"},
            "statistical_analysis": {},
            "ml_inference": {},
        }
        fake_analyzer = SimpleNamespace(
            analyze_video=lambda path, identify_user=True: fake_results,
        )

        request = _attach_session(
            factory.post(
                "/api/analyze",
                data=json.dumps({"filename": "sample.mp4", "identify_user": False}),
                content_type="application/json",
            )
        )
        remember_video_access(request, "sample.mp4")

        with (
            patch("server.webapp.analysis_views.UPLOAD_DIR", upload_dir),
            patch("server.webapp.analysis_views.RESULTS_DIR", results_dir),
            patch("server.webapp.analysis_views.ensure_analyzers", return_value=(fake_analyzer, None)),
            patch(
                "server.webapp.analysis_views.consume_usage",
                return_value=(
                    True,
                    {
                        "client_id": "ip:test",
                        "used_count": 1,
                        "remaining": None,
                        "limit": None,
                        "unlimited": True,
                    },
                ),
            ),
            patch("server.webapp.analysis_views._persist_result_record"),
        ):
            response = api_analyze(request)

        payload = _json(response)
        assert response.status_code == 200
        assert payload["video_info"]["duration"] == 5.0
        assert payload["input_processing"]["audio_removed"] is True
        assert "sample_results.json" in request.session["allowed_result_filenames"]
        assert (results_dir / "sample_results.json").exists()


def test_api_analyze_async_returns_job_contract() -> None:
    factory = RequestFactory()
    with TemporaryDirectory() as tmpdir:
        upload_dir = Path(tmpdir)
        (upload_dir / "sample.mp4").write_bytes(b"video")
        request = _attach_session(
            factory.post(
                "/api/analyze-async",
                data=json.dumps({"filename": "sample.mp4", "identify_user": False}),
                content_type="application/json",
            )
        )
        remember_video_access(request, "sample.mp4")

        with (
            patch("server.webapp.analysis_views.UPLOAD_DIR", upload_dir),
            patch("server.webapp.analysis_views.consume_usage", return_value=(True, {"client_id": "ip:test"})),
            patch("server.webapp.analysis_views.run_job"),
            patch("server.webapp.analysis_views.create_job", return_value={"job_id": "job123", "status": "queued", "job_type": "gait", "video_filename": "sample.mp4"}),
        ):
            response = api_analyze_async(request)

        payload = _json(response)
        assert response.status_code == 202
        assert payload["job_id"] == "job123"
        assert request.session["allowed_job_ids"] == ["job123"]


def test_api_analyze_symptoms_returns_gait_enriched_contract() -> None:
    factory = RequestFactory()

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        upload_dir = tmp / "uploads"
        results_dir = tmp / "results"
        upload_dir.mkdir()
        results_dir.mkdir()
        (upload_dir / "scan.mp4").write_bytes(b"video")

        track = SimpleNamespace(
            person_id="p1",
            duration_frames=60,
            start_frame=0,
            keypoints=[],
            confidence_scores=[],
        )
        video_info = {
            "fps": 30.0,
            "total_frames": 60,
            "duration": 2.0,
            "width": 320,
            "height": 240,
            "pose_backend": "mediapipe",
        }
        symptom_results = {
            "video_info": video_info,
            "n_persons": 1,
            "activity_summary": {"walking": 1.0, "resting": 0.0, "task": 0.0, "standing": 1.0},
            "persons": [
                {
                    "person_id": "p1",
                    "activity_segments": [],
                    "activity_breakdown": {"walking": 1.0, "standing": 1.0},
                    "symptoms": {"fog": {}},
                    "skeleton_track": {
                        "start_frame": 0,
                        "frame_stride": 1,
                        "n_frames": 0,
                        "frames": [],
                        "keypoints": [],
                        "pose_quality": [],
                    },
                }
            ],
            "analyzed_symptoms": ["fog"],
        }
        gait_results = {
            "video_info": video_info,
            "user": None,
            "pose_backend": "mediapipe",
            "gait_review_track": {"person_id": "p1"},
            "walking_detection": {
                "segments": [{"start_time": 0.2, "end_time": 1.2}],
                "walking_ratio": 0.5,
            },
            "analysis_results": [{"cadence": 120.0, "duration": 1.0}],
            "statistical_analysis": {},
            "preprocessing": {"detection_rate": 0.95, "mean_pose_quality": 0.85},
            "ml_inference": {},
            "summary": {
                "avg_pd_risk_score": 0.2,
                "avg_speed": 1.0,
                "avg_stride_length": 0.5,
                "avg_cadence": 120.0,
                "overall_classification": "Unknown",
            },
            "turn_detection": {"summary": {}},
            "turn_analysis": [],
            "turn_methodology": {},
            "activity_timeline": {"segments": [], "summary": {}},
        }
        fake_analyzer = SimpleNamespace(
            analyze_precomputed_pose=lambda **kwargs: gait_results,
        )
        fake_pd_symptoms = SimpleNamespace(
            tracker=SimpleNamespace(
                extract_tracks=lambda *args, **kwargs: ([track], video_info),
            ),
            analyze_tracks=lambda **kwargs: symptom_results,
        )

        request = _attach_session(
            factory.post(
                "/api/analyze-symptoms",
                data=json.dumps({"filename": "scan.mp4", "symptoms": None}),
                content_type="application/json",
            )
        )
        remember_video_access(request, "scan.mp4")

        with (
            patch("server.webapp.analysis_views.UPLOAD_DIR", upload_dir),
            patch("server.webapp.analysis_views.RESULTS_DIR", results_dir),
            patch(
                "server.webapp.analysis_views.ensure_analyzers",
                return_value=(fake_analyzer, fake_pd_symptoms),
            ),
            patch(
                "server.webapp.analysis_views.consume_usage",
                return_value=(
                    True,
                    {
                        "client_id": "ip:test",
                        "used_count": 1,
                        "remaining": None,
                        "limit": None,
                        "unlimited": True,
                    },
                ),
            ),
            patch("server.webapp.analysis_views._persist_result_record"),
        ):
            response = api_analyze_symptoms(request)

        payload = _json(response)
        assert response.status_code == 200
        assert payload["n_persons"] == 1
        assert payload["gait_analysis"]["pose_backend"] == "mediapipe"
        assert payload["gait_analysis"]["source_person_id"] == "p1"
        assert payload["persons"][0]["symptoms"]["fog"]["transitions_detected"] >= 0
        assert payload["input_processing"]["audio_removed"] is True
        assert "scan_symptoms.json" in request.session["allowed_result_filenames"]


def test_api_analyze_symptoms_async_returns_job_contract() -> None:
    factory = RequestFactory()
    with TemporaryDirectory() as tmpdir:
        upload_dir = Path(tmpdir)
        (upload_dir / "scan.mp4").write_bytes(b"video")
        request = _attach_session(
            factory.post(
                "/api/analyze-symptoms-async",
                data=json.dumps({"filename": "scan.mp4", "symptoms": None}),
                content_type="application/json",
            )
        )
        remember_video_access(request, "scan.mp4")

        with (
            patch("server.webapp.analysis_views.UPLOAD_DIR", upload_dir),
            patch("server.webapp.analysis_views.consume_usage", return_value=(True, {"client_id": "ip:test"})),
            patch("server.webapp.analysis_views.run_job"),
            patch("server.webapp.analysis_views.create_job", return_value={"job_id": "job999", "status": "queued", "job_type": "symptoms", "video_filename": "scan.mp4"}),
        ):
            response = api_analyze_symptoms_async(request)

        payload = _json(response)
        assert response.status_code == 202
        assert payload["job_id"] == "job999"


def test_api_job_status_requires_session_job_access() -> None:
    factory = RequestFactory()
    request = _attach_session(factory.get("/api/jobs/job123"))

    response = api_job_status(request, "job123")
    payload = _json(response)
    assert response.status_code == 403
    assert payload["error"] == "Forbidden"


def test_api_job_status_grants_result_access_on_success() -> None:
    factory = RequestFactory()
    request = _attach_session(factory.get("/api/jobs/job123"))
    request.session["allowed_job_ids"] = ["job123"]
    request.session.save()

    with patch(
        "server.webapp.analysis_views.get_job",
        return_value={
            "job_id": "job123",
            "status": "succeeded",
            "result_filename": "scan_symptoms.json",
            "video_filename": "scan.mp4",
        },
    ):
        response = api_job_status(request, "job123")

    payload = _json(response)
    assert response.status_code == 200
    assert payload["status"] == "succeeded"
    assert "scan_symptoms.json" in request.session["allowed_result_filenames"]


def test_api_result_file_normalizes_saved_payload() -> None:
    factory = RequestFactory()

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        upload_dir = tmp / "uploads"
        results_dir = tmp / "results"
        upload_dir.mkdir()
        results_dir.mkdir()
        (upload_dir / "saved.mp4").write_bytes(b"video")

        result_payload = {
            "activity_summary": {"walking": 1.0, "resting": 0.0, "task": 0.0, "standing": 1.0},
            "persons": [
                {
                    "person_id": "p1",
                    "activity_segments": [],
                    "activity_breakdown": {"walking": 1.0, "standing": 1.0},
                    "skeleton_track": {
                        "start_frame": 0,
                        "frame_stride": 1,
                        "n_frames": 1,
                        "keypoints": [[[0.0, 0.0, 1.0]]],
                        "pose_quality": [0.9],
                    },
                }
            ],
            "gait_analysis": {
                "source_person_id": "p1",
                "activity_timeline": {
                    "segments": [{"label": "walking", "start_time": 0.0, "end_time": 1.0}],
                    "summary": {"walking_time": 1.0, "standing_time": 1.0, "turning_time": 0.0},
                },
            },
        }
        (results_dir / "saved_symptoms.json").write_text(json.dumps(result_payload), encoding="utf-8")

        request = _attach_session(factory.get("/api/results/saved_symptoms.json"))
        remember_video_access(request, "saved.mp4")

        with (
            patch("server.webapp.results_views.RESULTS_DIR", results_dir),
            patch("server.webapp.results_views.UPLOAD_DIR", upload_dir),
        ):
            response = api_result_file(request, "saved_symptoms.json")

        payload = _json(response)
        assert response.status_code == 200
        assert payload["video_filename"] == "saved.mp4"
        assert payload["data"]["activity_schema"]["routing"]["kind"] == "symptom_routing"
        assert payload["data"]["persons"][0]["skeleton_track"]["frames"] == [[[0.0, 0.0, 1.0]]]


def test_api_results_lists_only_session_scoped_files() -> None:
    factory = RequestFactory()

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        upload_dir = tmp / "uploads"
        results_dir = tmp / "results"
        upload_dir.mkdir()
        results_dir.mkdir()
        (upload_dir / "saved.mp4").write_bytes(b"video")
        (results_dir / "saved_symptoms.json").write_text("{}", encoding="utf-8")

        request = _attach_session(factory.get("/api/results"))
        remember_video_access(request, "saved.mp4")

        with (
            patch("server.webapp.results_views.RESULTS_DIR", results_dir),
            patch("server.webapp.results_views.UPLOAD_DIR", upload_dir),
            patch(
                "server.webapp.results_views.AnalysisResult.objects.all",
                side_effect=Exception("db not available"),
            ),
        ):
            with patch("server.webapp.results_views.OperationalError", Exception), patch(
                "server.webapp.results_views.ProgrammingError", Exception
            ):
                response = api_results(request)

        payload = _json(response)
        assert response.status_code == 200
        assert len(payload) == 1
        assert payload[0]["filename"] == "saved_symptoms.json"
        assert payload[0]["video_filename"] == "saved.mp4"


def test_video_file_requires_session_access() -> None:
    factory = RequestFactory()
    request = _attach_session(factory.get("/videos/private.mp4"))

    response = video_file(request, "private.mp4")

    payload = _json(response)
    assert response.status_code == 403
    assert payload["error"] == "Forbidden"


def test_user_photo_requires_session_access() -> None:
    factory = RequestFactory()
    request = _attach_session(factory.get("/api/users/u1/photo"))

    response = user_photo(request, "u1")

    payload = _json(response)
    assert response.status_code == 403
    assert payload["error"] == "Forbidden"


def test_api_users_lists_only_session_owned_profiles() -> None:
    factory = RequestFactory()
    request = _attach_session(factory.get("/api/users"))
    request.session["allowed_user_ids"] = ["u2"]
    request.session.save()

    fake_analyzer = SimpleNamespace(
        list_users=lambda: [
            {"user_id": "u1", "name": "Hidden"},
            {"user_id": "u2", "name": "Visible"},
        ]
    )

    with patch("server.webapp.media_views.ensure_analyzers", return_value=(fake_analyzer, None)):
        response = api_users(request)

    payload = _json(response)
    assert response.status_code == 200
    assert payload == [{"user_id": "u2", "name": "Visible"}]


def test_api_analyze_returns_429_when_rate_limited() -> None:
    factory = RequestFactory()
    request = _attach_session(
        factory.post(
            "/api/analyze",
            data=json.dumps({"filename": "sample.mp4", "identify_user": False}),
            content_type="application/json",
        )
    )
    remember_video_access(request, "sample.mp4")

    with TemporaryDirectory() as tmpdir:
        upload_dir = Path(tmpdir)
        (upload_dir / "sample.mp4").write_bytes(b"video")
        with (
            patch("server.webapp.analysis_views.UPLOAD_DIR", upload_dir),
            patch(
                "server.webapp.analysis_views.consume_usage",
                return_value=(False, {"error": "Hourly analysis limit exceeded", "limit": 40, "remaining": 0}),
            ),
        ):
            response = api_analyze(request)

    payload = _json(response)
    assert response.status_code == 429
    assert payload["error"] == "Hourly analysis limit exceeded"
