#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DJANGO_SETTINGS_MODULE=server.config.settings.prod

uv run --extra pose --extra server python manage.py collectstatic --noinput >/dev/null

exec uv run --extra pose --extra server gunicorn \
  server.config.wsgi:application \
  --bind 0.0.0.0:8000 \
  --workers 2 \
  --timeout 300
