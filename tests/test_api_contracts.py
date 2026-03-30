"""API contract skeleton tests.

These tests intentionally validate response shape rather than the full runtime
behavior of the heavy analyzers. They act as a stable contract layer for the
payloads consumed by the Django UI.
"""

from __future__ import annotations

from server.webapp.activity_schema import normalize_activity_schema


def assert_keys_present(payload: dict, required_keys: set[str]) -> None:
    missing = required_keys.difference(payload.keys())
    assert not missing, f"Missing required keys: {sorted(missing)}"


def test_gait_analyze_contract_shape() -> None:
    """Guard the minimum payload shape for POST /api/analyze."""
    payload = {
        "video_info": {"duration": 12.5, "fps": 30.0},
        "user": None,
        "walking_detection": {"segments": [], "walking_ratio": 0.0},
        "preprocessing": {"detection_rate": 0.9, "mean_pose_quality": 0.8},
        "analysis_results": [],
        "summary": {"overall_classification": "Unknown"},
        "statistical_analysis": {},
        "ml_inference": {},
        "usage": {
            "client_id": "ip:test",
            "used_count": 0,
            "remaining": None,
            "limit": None,
            "unlimited": True,
        },
    }

    assert_keys_present(
        payload,
        {
            "video_info",
            "user",
            "walking_detection",
            "preprocessing",
            "analysis_results",
            "summary",
            "statistical_analysis",
            "ml_inference",
            "usage",
        },
    )


def test_full_scan_contract_shape() -> None:
    """Guard the minimum payload shape for POST /api/analyze-symptoms."""
    payload = {
        "video_info": {"duration": 12.5, "fps": 30.0},
        "n_persons": 1,
        "activity_summary": {"walking": 4.0, "resting": 2.0, "task": 1.0, "standing": 5.5},
        "persons": [
            {
                "person_id": "p1",
                "activity_segments": [],
                "activity_breakdown": {},
                "symptoms": {},
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
        "analyzed_symptoms": ["tremor", "bradykinesia", "posture", "fog"],
        "gait_analysis": {
            "pose_backend": "mediapipe",
            "video_info": {"duration": 12.5, "fps": 30.0},
            "user": None,
            "walking_detection": {"segments": [], "walking_ratio": 0.0},
            "analysis_results": [],
            "statistical_analysis": {},
            "ml_inference": {},
            "summary": {"overall_classification": "Unknown"},
            "activity_timeline": {"segments": [], "summary": {}},
            "turn_detection": {"summary": {}},
            "turn_analysis": [],
            "turn_methodology": {},
        },
        "usage": {
            "client_id": "ip:test",
            "used_count": 0,
            "remaining": None,
            "limit": None,
            "unlimited": True,
        },
    }

    assert_keys_present(
        payload,
        {
            "video_info",
            "n_persons",
            "activity_summary",
            "persons",
            "analyzed_symptoms",
            "gait_analysis",
            "usage",
        },
    )

    assert_keys_present(
        payload["gait_analysis"],
        {
            "pose_backend",
            "video_info",
            "user",
            "walking_detection",
            "analysis_results",
            "statistical_analysis",
            "ml_inference",
            "summary",
            "activity_timeline",
            "turn_detection",
            "turn_analysis",
            "turn_methodology",
        },
    )


def test_activity_schema_normalization_for_history_payload() -> None:
    """Saved result payloads should normalize to the UI's shared activity schema."""
    payload = {
        "activity_summary": {"walking": 4.0, "resting": 2.0, "task": 1.0, "standing": 5.5},
        "persons": [
            {
                "person_id": "p1",
                "activity_segments": [],
                "activity_breakdown": {"walking": 4.0, "standing": 5.5},
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
                "segments": [{"label": "walking", "start_time": 0.0, "end_time": 2.0}],
                "summary": {"walking_time": 2.0, "standing_time": 1.0, "turning_time": 0.0},
            },
        },
    }

    normalized = normalize_activity_schema(payload)

    assert "activity_schema" in normalized
    assert normalized["activity_schema"]["routing"]["kind"] == "symptom_routing"
    assert normalized["activity_schema"]["gait_phase"]["kind"] == "gait_phase"

    person = normalized["persons"][0]
    assert "activity_schema" in person
    assert person["activity_schema"]["owner_person_id"] == "p1"
    assert person["skeleton_track"]["frames"] == person["skeleton_track"]["keypoints"]
