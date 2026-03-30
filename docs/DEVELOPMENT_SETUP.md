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

Important behavior:

- `base.py` defines SQLite defaults
- `dev.py` overrides DB settings to PostgreSQL
- `prod.py` also uses PostgreSQL and enables production security/static settings

### Recommended Local Mode

For most local development, choose one mode explicitly instead of relying on implicit defaults.

#### Option A: Local SQLite-oriented workflow

```bash
export DJANGO_SETTINGS_MODULE=server.config.settings.base
python manage.py migrate
python manage.py runserver
```

Use this when:

- you are working on UI, routing, or payload shape
- you do not need the Postgres runtime path
- you want the simplest local startup

#### Option B: Local/Postgres-aligned workflow

```bash
export DJANGO_SETTINGS_MODULE=server.config.settings.dev
export POSTGRES_DB=mydb
export POSTGRES_USER=admin
export POSTGRES_PASSWORD=...
export POSTGRES_HOST=postgres
export POSTGRES_PORT=5432
python manage.py migrate
python manage.py runserver
```

Use this when:

- you want behavior closer to the current dev/prod DB topology
- you already have a running Postgres instance

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
- `POST /api/analyze-symptoms`
- `GET /api/reference-data`
- `GET /api/results`
- `GET /api/results/<filename>`
- `GET /api/status`

Route definitions live in:

- [`server/webapp/urls.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/urls.py)

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
