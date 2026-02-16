from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("api/status", views.api_status, name="api_status"),
    path("api/upload", views.api_upload, name="api_upload"),
    path("api/analyze", views.api_analyze, name="api_analyze"),
    path("api/analyze-symptoms", views.api_analyze_symptoms, name="api_analyze_symptoms"),
    path("api/reference-data", views.api_reference_data, name="api_reference_data"),
    path("api/register-user", views.api_register_user, name="api_register_user"),
    path("api/users", views.api_users, name="api_users"),
    path("api/users/<str:user_id>/photo", views.user_photo, name="user_photo"),
    path("videos/<str:filename>", views.video_file, name="video_file"),
    path("favicon.ico", views.favicon, name="favicon"),
    path("status", views.api_status, name="status"),
    path("upload", views.api_upload, name="upload"),
    path("analyze", views.api_analyze, name="analyze"),
    path("analyze-symptoms", views.api_analyze_symptoms, name="analyze_symptoms"),
    path("reference-data", views.api_reference_data, name="reference_data"),
    path("register-user", views.api_register_user, name="register_user"),
    path("users", views.api_users, name="users"),
    path("users/<str:user_id>/photo", views.user_photo, name="user_photo_legacy"),
]
