# Development Setup

Last verified: 2026-03-30

## Purpose

This document defines the practical local development path for this repository.

The repository contains two distinct runtimes:

1. Python package development under `src/research_automation/`
2. Django web service development under `server/`

## Baseline Environment

Recommended baseline:

- Python 3.11+
- `uv` for dependency management

Install the baseline development environment:

```bash
uv sync --extra dev
```

For web analysis work, install the Django and pose-related extras used by the service:

```bash
uv sync --extra dev --extra server --extra pose
```

Optional extras:

- `--extra cv` for MediaPipe-related CV support
- `--extra ecg` for ECG tooling
- `--extra audio` for audio tooling

## Main Entry Points

### Package CLI

The package CLI entrypoint is:

- `research`

Examples:

```bash
research --help
research lit search "Parkinson gait video" --source pubmed --limit 5
research lit download --arxiv 2311.09890
```

### Django Service

The Django entrypoint is:

- [`manage.py`](/workspace/video-based-PD-risk-analyzer/manage.py)

Current default:

- `manage.py` sets `DJANGO_SETTINGS_MODULE=server.config.settings.dev`

That means a plain `python manage.py runserver` uses the development settings module unless explicitly overridden.

## Django Settings Modes

Verified settings modules:

- base: [`server/config/settings/base.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/base.py)
- dev: [`server/config/settings/dev.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/dev.py)
- prod: [`server/config/settings/prod.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/prod.py)
- local_sqlite: [`server/config/settings/local_sqlite.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/local_sqlite.py)
- test: [`server/config/settings/test.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/test.py)

Important behavior:

- `base.py` now defines the shared Postgres-first baseline
- `dev.py` and `prod.py` inherit that Postgres baseline
- `local_sqlite.py` is an explicit opt-in fallback for no-DB local work
- `test.py` is SQLite-backed and reserved for automated tests

### Recommended Local Mode

For most local development, choose one mode explicitly instead of relying on implicit defaults.

#### Option A: Local Postgres workflow

```bash
export DJANGO_SETTINGS_MODULE=server.config.settings.dev
export POSTGRES_DB=mydb
export POSTGRES_USER=admin
export POSTGRES_PASSWORD=...
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
python manage.py migrate
python manage.py runserver
```

Use this when:

- you are doing normal application development
- you want login/session behavior to match product configuration
- you want one database topology across local, dev, and prod

#### Option B: Explicit SQLite fallback

```bash
export DJANGO_SETTINGS_MODULE=server.config.settings.local_sqlite
python manage.py migrate
python manage.py runserver
```

Use this when:

- Postgres is temporarily unavailable
- you need a disposable no-dependency local sandbox

This fallback should not be treated as the primary product validation path.

## Runtime Directories

The Django app uses runtime paths under `server/runtime/`.

Main directories:

- uploads: `server/runtime/uploads`
- results: `server/runtime/results`
- data: `server/runtime/data`
- profiles: `server/runtime/data/profiles`

These are derived through:

- [`server/webapp/app_settings.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/app_settings.py)
- [`server/webapp/runtime.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/runtime.py)

## API And Route Surface

The app serves both explicit `/api/*` paths and legacy aliases without `/api`.

Canonical routes for new development:

- `POST /api/upload`
- `POST /api/analyze`
- `POST /api/analyze-async`
- `POST /api/analyze-symptoms`
- `POST /api/analyze-symptoms-async`
- `GET /api/jobs/<job_id>`
- `GET /api/reference-data`
- `GET /api/results`
- `GET /api/results/<filename>`
- `GET /api/status`
- `GET /api/auth/session`
- `GET /login`
- `POST /logout`
- `GET /api/users`
- `GET /api/users/<user_id>/photo`

Route definitions live in:

- [`server/webapp/urls.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/urls.py)

## Session-Scoped Access

The web product now treats uploaded media and generated artifacts as browser-session scoped by default.

Current behavior:

- `/api/upload` grants the current session access to the normalized analysis video
- `/api/analyze*` only accepts filenames already granted to the current session
- `/api/jobs/<job_id>` only exposes jobs created by the current session
- `/api/results` and `/api/results/<filename>` only expose results linked to session-owned videos or previously unlocked results
- `/api/users` only lists users created in the current session
- `/videos/<filename>` and `/api/users/<user_id>/photo` return `403` outside the allowed session

This is a product hardening measure for privacy and demo isolation, not a full authentication system.

## Authentication And Session Storage

The app now uses Django's built-in authentication stack.

Current behavior:

- the main UI at `/` requires login and redirects anonymous users to `/login`
- `/register` creates inactive users and sends verification emails
- `/verify-email` activates accounts from signed email links
- protected API routes return `401` JSON instead of redirecting
- session persistence does not require a cache database
- on Postgres-backed settings, auth/session rows are stored in the primary Postgres database
- on `test.py` and `local_sqlite.py`, auth/session rows are stored in SQLite
- the default session backend is Django's DB session engine
- default login sessions expire on browser close unless `remember me` is checked
- persistent sessions use `SESSION_COOKIE_AGE` and renew on activity because `SESSION_SAVE_EVERY_REQUEST=True`
- session cookies are `HttpOnly` and `SameSite=Lax` by default

Email delivery:

- `test.py` uses Django's in-memory email backend
- the default runtime uses Django's console email backend unless SMTP env vars are configured

Rate limiting:

- auth and analysis throttling are stored in the primary database via `RateLimitEvent`
- no Redis or cache DB is required for the current abuse-control policy

## Pose Backend Selection

Pose backend selection is centralized in:

- [`server/webapp/analysis_config.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/analysis_config.py)

Current default:

- `mediapipe`

Override example:

```bash
export POSE_BACKEND=rtmw
```

Accepted values:

- `auto`
- `mediapipe`
- `rtmw`
- `none`

## Verification Commands

Useful local checks:

```bash
python -m compileall src tests
pytest -q
python manage.py check
python manage.py migrate --plan
```

## Experiment Tracking

The experiment module uses MLflow.

Current default local behavior:

- the local tracking root remains `mlruns/`
- the backend is normalized to a local SQLite database at `mlruns/mlflow.db`
- artifacts are stored under `mlruns/artifacts/`

This avoids the deprecated filesystem tracking backend while keeping the familiar local folder layout.

## Current Verification Status

Current verified state in this environment:

- `python -m pytest -q` passes
- the previous Anthropic dependency path has been removed
- the local MLflow default no longer uses the deprecated filesystem tracking backend
- Postgres is the primary Django database path for app development
- SQLite remains only as an explicit fallback (`local_sqlite`) and for automated tests (`test`)
