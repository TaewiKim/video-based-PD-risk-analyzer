from .app_settings import app_settings


RUNTIME_DIR = app_settings.runtime_dir
UPLOAD_DIR = app_settings.upload_dir
RESULTS_DIR = app_settings.results_dir
DATA_DIR = app_settings.data_dir
PROFILE_DIR = app_settings.profile_dir
ALLOWED_EXTENSIONS = set(app_settings.allowed_video_extensions)


for folder in [RUNTIME_DIR, UPLOAD_DIR, RESULTS_DIR, DATA_DIR, PROFILE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
