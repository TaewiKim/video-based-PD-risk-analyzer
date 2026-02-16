# Research Automation

Video-based health monitoring research automation toolkit.

## Features

- **Literature Module**: Search, download, and summarize research papers from PubMed, arXiv, Semantic Scholar
- **Collection Module**: YouTube video search/download, quality assessment, dataset management
- **Clinical Scales**: MDS-UPDRS, Hoehn & Yahr, House-Brackmann and more
- **ECG Processing**: Load and analyze ECG signals (optional)

## Installation

```bash
# Using uv
uv sync

# With MediaPipe for full pose/face detection
uv sync --extra cv

# Development
uv sync --extra dev
```

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
export ANTHROPIC_API_KEY="your-key"
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
- `docs/RESEARCH_NOTE_PD_RISK_2026-02-15.md` (PD risk 모델 개선 실험 노트)

## Project Structure

```
research-automation/
├── src/research_automation/
│   ├── core/           # Config, database, storage, Claude API
│   ├── literature/     # Paper search, download, summarization
│   ├── collection/     # Video collection, quality, datasets
│   └── cli.py          # Typer CLI application
├── config/
│   ├── settings.yaml
│   └── search_queries.yaml
└── tests/
```

## License

MIT
