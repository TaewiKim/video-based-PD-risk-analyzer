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


def postgres_database_config(*, db_name: str | None = None) -> dict:
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": db_name or os.getenv("POSTGRES_DB", "mydb"),
        "USER": os.getenv("POSTGRES_USER", "admin"),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
        "HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "PORT": os.getenv("POSTGRES_PORT", "5432"),
        "CONN_MAX_AGE": int(os.getenv("POSTGRES_CONN_MAX_AGE", "60")),
    }

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

DATABASES = {"default": postgres_database_config()}

AUTH_PASSWORD_VALIDATORS = []
LOGIN_URL = "/login"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/login"
SESSION_ENGINE = "django.contrib.sessions.backends.db"
SESSION_COOKIE_NAME = os.getenv("DJANGO_SESSION_COOKIE_NAME", "pdgait_sessionid")
SESSION_COOKIE_AGE = int(os.getenv("DJANGO_SESSION_COOKIE_AGE", str(60 * 60 * 8)))
SESSION_EXPIRE_AT_BROWSER_CLOSE = os.getenv("DJANGO_SESSION_EXPIRE_AT_BROWSER_CLOSE", "1") == "1"
SESSION_SAVE_EVERY_REQUEST = os.getenv("DJANGO_SESSION_SAVE_EVERY_REQUEST", "1") == "1"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = os.getenv("DJANGO_SESSION_COOKIE_SAMESITE", "Lax")
SESSION_COOKIE_SECURE = os.getenv("DJANGO_SESSION_COOKIE_SECURE", "0") == "1"
CSRF_COOKIE_HTTPONLY = False
CSRF_COOKIE_SAMESITE = os.getenv("DJANGO_CSRF_COOKIE_SAMESITE", "Lax")
CSRF_COOKIE_SECURE = os.getenv("DJANGO_CSRF_COOKIE_SECURE", "0") == "1"

EMAIL_BACKEND = os.getenv("DJANGO_EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend")
EMAIL_HOST = os.getenv("DJANGO_EMAIL_HOST", "")
EMAIL_PORT = int(os.getenv("DJANGO_EMAIL_PORT", "587"))
EMAIL_HOST_USER = os.getenv("DJANGO_EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.getenv("DJANGO_EMAIL_HOST_PASSWORD", "")
EMAIL_USE_TLS = os.getenv("DJANGO_EMAIL_USE_TLS", "1") == "1"
DEFAULT_FROM_EMAIL = os.getenv("DJANGO_DEFAULT_FROM_EMAIL", "no-reply@pd-gait.local")

EMAIL_VERIFICATION_TTL_HOURS = int(os.getenv("EMAIL_VERIFICATION_TTL_HOURS", "24"))
AUTH_REGISTER_PER_IP_HOUR = int(os.getenv("AUTH_REGISTER_PER_IP_HOUR", "5"))
AUTH_REGISTER_PER_EMAIL_HOUR = int(os.getenv("AUTH_REGISTER_PER_EMAIL_HOUR", "3"))
AUTH_RESEND_PER_EMAIL_HOUR = int(os.getenv("AUTH_RESEND_PER_EMAIL_HOUR", "5"))
AUTH_LOGIN_WINDOW_MINUTES = int(os.getenv("AUTH_LOGIN_WINDOW_MINUTES", "15"))
AUTH_LOGIN_PER_IP_WINDOW = int(os.getenv("AUTH_LOGIN_PER_IP_WINDOW", "10"))
AUTH_LOGIN_PER_USER_WINDOW = int(os.getenv("AUTH_LOGIN_PER_USER_WINDOW", "8"))
AUTH_VERIFY_PER_IP_HOUR = int(os.getenv("AUTH_VERIFY_PER_IP_HOUR", "20"))
ANALYSIS_PER_USER_HOUR = int(os.getenv("ANALYSIS_PER_USER_HOUR", "12"))
ANALYSIS_PER_USER_DAY = int(os.getenv("ANALYSIS_PER_USER_DAY", "40"))

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
