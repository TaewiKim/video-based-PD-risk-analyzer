import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    for env_path in [BASE_DIR / ".env", BASE_DIR.parent / ".env"]:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not os.environ.get(key):
                os.environ[key] = value


_load_env_file()

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "dev-only-secret-key-change-me-please-override-with-a-long-random-production-secret",
)
DEBUG = True
ALLOWED_HOSTS = ["*"]
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
RESULTS_DIR = RUNTIME_DIR / "results"
DATA_DIR = RUNTIME_DIR / "data"
PROFILE_DIR = DATA_DIR / "profiles"
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "webm", "mkv"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "server.webapp",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "server.config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "server.config.wsgi.application"
ASGI_APPLICATION = "server.config.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
