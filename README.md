# Research Automation

Video-based health monitoring research automation toolkit.

## Features

- **Literature Module**: Search and download research papers from PubMed, arXiv, Semantic Scholar
- **Collection Module**: YouTube video search/download, quality assessment, dataset management
- **Clinical Scales**: MDS-UPDRS, Hoehn & Yahr, House-Brackmann and more
- **ECG Processing**: Load and analyze ECG signals (optional)

## Installation

```bash
# Using uv
uv sync

# With pose backends for web analysis
uv sync --extra cv
uv sync --extra pose

# Development
uv sync --extra dev

# Django service runtime
uv sync --extra pose --extra server
```

## Django Service

The web UI and API are now embedded in the Django app under `server/`.

```bash
# Apply migrations
python manage.py migrate

# Start development server
python manage.py runserver

# Or use the helper script
./scripts/run_django_server.sh
```

Open `http://127.0.0.1:8000`.

Production notes:

- `scripts/run_django_server.sh` runs `collectstatic` and starts Gunicorn with the production settings module.
- Static files are served in production via WhiteNoise.
- API surface summary: [`docs/API_REFERENCE.md`](/workspace/video-based-PD-risk-analyzer/docs/API_REFERENCE.md)
- Local development setup: [`docs/DEVELOPMENT_SETUP.md`](/workspace/video-based-PD-risk-analyzer/docs/DEVELOPMENT_SETUP.md)

## Quick Start

```bash
# Initialize configuration
research config init

# Search literature
research lit search "Parkinson gait video" --source pubmed --limit 10

# Download paper
research lit download --arxiv 2311.09890

# Search YouTube videos
research collect youtube "Parkinson tremor" --max 5

# Check video quality
research collect quality-check data/videos/raw/

# List available datasets
research collect dataset list

# View clinical scales
research collect scale list
research collect scale show MDS-UPDRS-III
```

## Configuration

Set environment variables or edit `config/settings.yaml`:

```bash
export PUBMED_EMAIL="your@email.com"
```

## Dataset Download Links

Large datasets are **not** committed to this repository. Download them separately:

- CARE-PD (Hugging Face): `https://huggingface.co/datasets/vida-adl/CARE-PD/tree/main`
- CARE-PD official code: `https://github.com/TaatiTeam/CARE-PD`
- 3DGait demo/reference repo: `https://github.com/lisqzqng/Video-based-gait-analysis-for-dementia`

After download, place files under `data/datasets/` (already ignored by git).

## CARE-PD Reproducibility

RF baseline vs CARE-PD official 코드 직접 실행/비교 절차는 아래 문서에 정리되어 있습니다.

- `docs/CAREPD_REPRO.md`
- `docs/PROJECT_FULL_DOCUMENTATION.md` (프로젝트 개념/문헌/아키텍처/분석 파이프라인/평가/결과 통합 문서)
- `docs/MODELS_AND_FORMULAS.md` (사용 모델, 문헌 cutoff, 통계식, 참고 문헌 정리)
- `docs/RESEARCH_NOTE_PD_RISK_2026-02-15.md` (PD risk 모델 개선 실험 노트)

## Project Structure

```
research-automation/
├── src/research_automation/
│   ├── core/           # Config, database, storage
│   ├── literature/     # Paper search, download, extraction
│   ├── collection/     # Video collection, quality, datasets
│   └── cli.py          # Typer CLI application
├── config/
│   ├── settings.yaml
│   └── search_queries.yaml
├── server/
│   ├── webapp/         # Django API, embedded UI, analysis services
│   └── runtime/        # Uploaded videos, models, saved results
├── manage.py           # Django entrypoint
└── tests/
```

## License

MIT
