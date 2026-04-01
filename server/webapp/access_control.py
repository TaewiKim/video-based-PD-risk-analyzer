from __future__ import annotations

from collections.abc import Iterable


SESSION_ALLOWED_VIDEOS = "allowed_video_filenames"
SESSION_ALLOWED_RESULTS = "allowed_result_filenames"
SESSION_ALLOWED_USERS = "allowed_user_ids"
SESSION_ALLOWED_JOBS = "allowed_job_ids"


def _get_session_list(request, key: str) -> list[str]:
    values = request.session.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if v]


def _store_session_list(request, key: str, values: Iterable[str]) -> None:
    request.session[key] = sorted({str(v) for v in values if v})
    request.session.modified = True


def remember_value(request, key: str, value: str) -> None:
    values = _get_session_list(request, key)
    values.append(value)
    _store_session_list(request, key, values)


def remember_video_access(request, filename: str) -> None:
    remember_value(request, SESSION_ALLOWED_VIDEOS, filename)


def remember_result_access(request, filename: str) -> None:
    remember_value(request, SESSION_ALLOWED_RESULTS, filename)


def remember_user_access(request, user_id: str) -> None:
    remember_value(request, SESSION_ALLOWED_USERS, user_id)


def remember_job_access(request, job_id: str) -> None:
    remember_value(request, SESSION_ALLOWED_JOBS, job_id)


def can_access_video(request, filename: str) -> bool:
    return filename in set(_get_session_list(request, SESSION_ALLOWED_VIDEOS))


def can_access_result(request, filename: str) -> bool:
    return filename in set(_get_session_list(request, SESSION_ALLOWED_RESULTS))


def can_access_user(request, user_id: str) -> bool:
    return user_id in set(_get_session_list(request, SESSION_ALLOWED_USERS))


def can_access_job(request, job_id: str) -> bool:
    return job_id in set(_get_session_list(request, SESSION_ALLOWED_JOBS))


def allowed_videos(request) -> set[str]:
    return set(_get_session_list(request, SESSION_ALLOWED_VIDEOS))


def allowed_results(request) -> set[str]:
    return set(_get_session_list(request, SESSION_ALLOWED_RESULTS))


def allowed_users(request) -> set[str]:
    return set(_get_session_list(request, SESSION_ALLOWED_USERS))
