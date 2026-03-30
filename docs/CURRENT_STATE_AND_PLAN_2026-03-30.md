# Current State And Plan

Last verified: 2026-03-30
Repository: `video-based-PD-risk-analyzer`

## 1. What Exists Now

This repository currently has two product layers in one codebase.

1. A Python package under `src/research_automation/` for:
   - literature search/download/summarization
   - dataset and video collection
   - experiment tracking and report generation
   - video and gait feature extraction pipelines
2. A Django service under `server/` for:
   - video upload and storage
   - gait-only analysis
   - multi-person symptom analysis
   - result persistence and browser-based review UI

The main runtime entrypoints verified in code are:

- CLI: `research` via [`pyproject.toml`](/workspace/video-based-PD-risk-analyzer/pyproject.toml)
- Django: [`manage.py`](/workspace/video-based-PD-risk-analyzer/manage.py)
- Root URL router: [`server/config/urls.py`](/workspace/video-based-PD-risk-analyzer/server/config/urls.py)
- App routes: [`server/webapp/urls.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/urls.py)

## 2. Verified Architecture

### Package layer

Verified top-level package modules:

- `analysis`
- `augmentation`
- `collection`
- `core`
- `experiment`
- `literature`
- `models`
- `pipeline`
- `report`
- `submission`
- `visualization`

The CLI in [`src/research_automation/cli.py`](/workspace/video-based-PD-risk-analyzer/src/research_automation/cli.py) already exposes grouped commands for:

- `config`
- `lit`
- `collect`
- `pipeline`
- `experiment`
- `report`

### Web layer

The Django app exposes both `/api/*` and legacy non-prefixed endpoints.

Verified API routes:

- `POST /api/upload`
- `POST /api/analyze`
- `POST /api/analyze-symptoms`
- `GET /api/reference-data`
- `GET /api/results`
- `GET /api/results/<filename>`
- `GET /api/status`
- `POST /api/register-user`
- `GET /api/users`
- `GET /api/users/<user_id>/photo`
- `GET /videos/<filename>`

The current analysis orchestration is split across:

- [`server/webapp/analysis_views.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/analysis_views.py)
- [`server/webapp/services/smart_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/smart_analyzer.py)
- [`server/webapp/services/pd_symptoms_analyzer.py`](/workspace/video-based-PD-risk-analyzer/server/webapp/services/pd_symptoms_analyzer.py)

## 3. Environment And Runtime Facts

### Dependency model

The project is configured through [`pyproject.toml`](/workspace/video-based-PD-risk-analyzer/pyproject.toml) with:

- base Python requirement `>=3.11`
- core runtime dependencies for CV, scientific Python, Django, PostgreSQL, and ML
- optional extras: `dev`, `cv`, `pose`, `ecg`, `audio`, `server`

### Django settings

Verified settings modules:

- base: [`server/config/settings/base.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/base.py)
- dev: [`server/config/settings/dev.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/dev.py)
- prod: [`server/config/settings/prod.py`](/workspace/video-based-PD-risk-analyzer/server/config/settings/prod.py)

Important current behavior:

- [`manage.py`](/workspace/video-based-PD-risk-analyzer/manage.py) defaults to `server.config.settings.dev`
- `dev.py` uses PostgreSQL by default
- `base.py` still defines SQLite defaults
- timezone is set to `Asia/Seoul`
- `.env` loading is implemented inside `base.py`

This means local developer startup can diverge depending on whether they use `manage.py`, direct Django settings overrides, or existing Postgres runtime state.

## 4. Current Validation Status

### Tests

`pytest -q` does not currently run successfully in this environment.

Observed current blocker:

- import failure at test collection time
- current environment does not have `sqlalchemy` available

Recent change already applied:

- the previous `anthropic` / Claude dependency path was removed from the repository
- package import is no longer blocked by optional LLM code

Current impact:

- tests are now blocked by missing baseline package dependencies, not by optional LLM coupling

### Git worktree

The repository is not clean.

Observed modified or untracked areas include:

- Django settings
- Django models/views/templates
- SQLite DB files
- runtime uploads and saved results
- Postgres runtime data

This should be treated as an active working tree, not a clean baseline.

## 5. Documentation Gaps Found

The repository already contains strong documentation coverage, but not all documents appear aligned with the current file layout.

Examples observed during verification:

- [`docs/PROJECT_FULL_DOCUMENTATION.md`](/workspace/video-based-PD-risk-analyzer/docs/PROJECT_FULL_DOCUMENTATION.md) references files such as `server/webapp/views.py` and template/service paths that do not match the current split into `analysis_views.py`, `media_views.py`, `page_views.py`, and `results_views.py`
- the route examples in existing docs still describe non-`/api` endpoints as the primary contract, while the current router serves both
- runtime and settings behavior are documented incompletely relative to the current Postgres-first dev setup

## 6. Practical Assessment

### Strengths

- the codebase already has clear separation between research toolkit and web demo
- the Django service has route modularization instead of a single large views module
- the analysis pipeline already persists JSON result artifacts
- there is an existing test suite across core, literature, collection, pipeline, experiment, and report modules
- project docs already cover API, runtime fallback, activity schema, and reproducibility topics

### Risks

- baseline dependencies are not installed in the current environment
- local startup behavior is ambiguous because settings defaults do not match each other
- runtime artifacts and database files live inside the repo tree, which increases noise and drift
- historical docs are at risk of becoming misleading because architecture moved faster than documentation updates
- analysis code in `smart_analyzer.py` and `pd_symptoms_analyzer.py` is large and likely needs modular extraction for maintainability

## 7. Recommended Development Plan

### Phase 1: Stabilize The Developer Baseline

Goal: make the repository predictable to install, test, and run.

Actions:

1. Make `pytest` run with the documented baseline development dependencies.
2. Keep optional integrations out of package import time.
3. Define one default local DB path for development and document the override path for Postgres.
4. Move or ignore runtime artifacts more aggressively so the repo reflects code changes, not generated state.

Success criteria:

- `pytest` collects and runs core unit tests in a fresh environment
- `uv sync --extra dev` is enough for baseline development
- local Django startup path is documented and reproducible

### Phase 2: Reconcile Code And Docs

Goal: make the documentation trustworthy again.

Actions:

1. Update the architecture document to match the current split view modules and real routes.
2. Add a short "how to run locally" document covering CLI, Django, DB mode, and pose backend options.
3. Separate stable API contract docs from implementation notes.
4. Mark older documents with verification dates so drift is visible.

Success criteria:

- one current architecture document
- one current developer setup document
- one current API contract document

### Phase 3: Harden The Analysis Service

Goal: reduce fragility in the web analysis path.

Actions:

1. Extract smaller service modules from the two large analyzers.
2. Add schema-level tests for `/api/analyze` and `/api/analyze-symptoms`.
3. Add fixture-based tests for saved-result normalization and history reload behavior.
4. Introduce structured logging around upload, analysis, persistence, and failures.

Success criteria:

- deterministic API schema tests
- clearer failure diagnostics
- lower risk when changing UI or payload shape

### Phase 4: Improve Model And Product Direction

Goal: move from demo utility toward a robust PD-risk analysis platform.

Actions:

1. Define a versioned result schema for gait, symptoms, and activity outputs.
2. Add dataset registry and experiment metadata that connect web inference to reproducible offline experiments.
3. Establish benchmark datasets and acceptance metrics for gait and symptom detection quality.
4. Separate literature-derived rule signals from learned model signals more explicitly in both API and UI.

Success criteria:

- reproducible offline-to-online model story
- versioned outputs
- measurable release criteria for analysis quality

## 8. Highest-Value Next Steps

If work starts immediately, the best order is:

1. Install and verify the baseline dev dependencies used by the test suite.
2. Add a minimal CI baseline: lint plus a dependency-light pytest subset.
3. Normalize local runtime setup for Django and database selection.
4. Refresh the main architecture and setup docs after the code baseline is stable.
5. Add API contract tests around `analyze` and `analyze-symptoms`.

## 9. Suggested Ownership Split

For parallel execution, this repository can be split into three tracks:

- Platform track: packaging, settings, testability, CI
- Analysis track: gait and symptom service refactor, schema validation
- Product track: UI/data contract cleanup, result history, documentation

This split will reduce merge conflicts because the write surfaces are naturally separated.
