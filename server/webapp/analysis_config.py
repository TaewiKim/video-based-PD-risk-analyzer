import os


DEFAULT_POSE_BACKEND = "mediapipe"


def get_pose_backend(explicit: str | None = None) -> str:
    candidate = (explicit or os.getenv("POSE_BACKEND") or DEFAULT_POSE_BACKEND).strip().lower()
    if candidate in {"auto", "mediapipe", "rtmw", "none"}:
        return candidate
    return DEFAULT_POSE_BACKEND
