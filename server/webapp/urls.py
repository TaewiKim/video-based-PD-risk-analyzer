from django.urls import path

from .auth_views import (
    api_auth_session,
    login_view,
    logout_view,
    register_view,
    resend_verification_view,
    verify_email_view,
)
from .analysis_views import (
    api_analyze,
    api_analyze_async,
    api_analyze_symptoms,
    api_analyze_symptoms_async,
    api_job_status,
    api_reference_data,
)
from .media_views import api_register_user, api_upload, api_users, user_photo, video_file
from .page_views import favicon, index
from .results_views import api_result_file, api_results, api_status


urlpatterns = [
    path("login", login_view, name="login"),
    path("register", register_view, name="register"),
    path("resend-verification", resend_verification_view, name="resend_verification"),
    path("verify-email", verify_email_view, name="verify_email"),
    path("logout", logout_view, name="logout"),
    path("api/auth/session", api_auth_session, name="api_auth_session"),
    path("", index, name="index"),
    path("api/results", api_results, name="api_results"),
    path("api/results/<str:filename>", api_result_file, name="api_result_file"),
    path("api/status", api_status, name="api_status"),
    path("api/upload", api_upload, name="api_upload"),
    path("api/analyze", api_analyze, name="api_analyze"),
    path("api/analyze-async", api_analyze_async, name="api_analyze_async"),
    path("api/analyze-symptoms", api_analyze_symptoms, name="api_analyze_symptoms"),
    path("api/analyze-symptoms-async", api_analyze_symptoms_async, name="api_analyze_symptoms_async"),
    path("api/jobs/<str:job_id>", api_job_status, name="api_job_status"),
    path("api/reference-data", api_reference_data, name="api_reference_data"),
    path("api/register-user", api_register_user, name="api_register_user"),
    path("api/users", api_users, name="api_users"),
    path("api/users/<str:user_id>/photo", user_photo, name="user_photo"),
    path("videos/<str:filename>", video_file, name="video_file"),
    path("favicon.ico", favicon, name="favicon"),
    path("status", api_status, name="status"),
    path("results", api_results, name="results"),
    path("results/<str:filename>", api_result_file, name="result_file"),
    path("upload", api_upload, name="upload"),
    path("analyze", api_analyze, name="analyze"),
    path("analyze-async", api_analyze_async, name="analyze_async"),
    path("analyze-symptoms", api_analyze_symptoms, name="analyze_symptoms"),
    path("analyze-symptoms-async", api_analyze_symptoms_async, name="analyze_symptoms_async"),
    path("jobs/<str:job_id>", api_job_status, name="job_status"),
    path("reference-data", api_reference_data, name="reference_data"),
    path("register-user", api_register_user, name="register_user"),
    path("users", api_users, name="users"),
    path("users/<str:user_id>/photo", user_photo, name="user_photo_legacy"),
]
