from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock, Thread
from typing import Any, Callable
from uuid import uuid4


_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(*, job_type: str, video_filename: str | None = None) -> dict[str, Any]:
    job_id = uuid4().hex
    job = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "queued",
        "video_filename": video_filename,
        "result_filename": None,
        "result_type": None,
        "error": None,
        "submitted_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "progress_message": "Queued",
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return deepcopy(job)


def update_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        job.update(fields)
        return deepcopy(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return deepcopy(job) if job is not None else None


def run_job(job_id: str, worker: Callable[[], dict[str, Any]]) -> None:
    def _target() -> None:
        update_job(
            job_id,
            status="running",
            started_at=_utc_now(),
            progress_message="Running analysis",
        )
        try:
            result = worker()
            update_job(
                job_id,
                status="succeeded",
                finished_at=_utc_now(),
                progress_message="Complete",
                result_filename=result.get("result_filename"),
                result_type=result.get("result_type"),
                video_filename=result.get("video_filename"),
            )
        except Exception as exc:
            update_job(
                job_id,
                status="failed",
                finished_at=_utc_now(),
                progress_message="Failed",
                error=str(exc),
            )

    thread = Thread(target=_target, daemon=True)
    thread.start()
