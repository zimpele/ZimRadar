# ZimRadar

**Multi-agent AI system for climate and infrastructure risk assessment from satellite imagery.**

ZimRadar ingests Sentinel-2 satellite tiles, NOAA weather data, and FEMA disaster records to generate per-region risk scores for floods, wildfires, and land-use degradation. Enter coordinates or a US county — get a structured risk report with satellite overlays, trend forecasts, and traceable citations.

---

## What's implemented

### Phase 1 — Foundation
- **Sentinel-2 ingestion** via `sentinelsat` — nightly tile fetches for configured regions
- **NOAA weather ingestion** — daily precipitation, temperature, soil moisture
- **FEMA disaster records** — bulk sync + delta updates
- **SegFormer segmentation** — classifies tiles into water / vegetation / urban / bare soil / burn scar
- **pgvector** — embeddings for FEMA reports and satellite tiles
- **Prefect** — pipeline scheduling and observability
- **Streamlit dashboard** — live at `http://187.77.95.56:8501`

### Phase 2 — Intelligence
- **ZoeDepth terrain analysis** — identifies flood-accumulation zones from monocular depth estimation (lowest 10% elevation)
- **Chronos forecasting** — 30/60/90-day flood and fire risk flags from probabilistic precipitation and temperature forecasts
- **XGBoost risk classifier** — 4-tier risk label (low / moderate / high / critical) from engineered features
- **RAG retriever** — pgvector similarity search over FEMA reports with CrossEncoder reranking and metadata filters

### Coming in Phase 3
- LangGraph multi-agent orchestration (Ingest → Analysis → Report → Validator)
- LLM-generated narrative risk reports with inline citations
- Validator Agent with LLM-as-judge factuality scoring
- FastAPI REST endpoints (`POST /assess`, `GET /report/{id}`)
- PDF report export

---

## Architecture

```
Sentinel-2 / NOAA / FEMA / OSM
         │
         ▼
   Prefect Pipeline
         │
    ┌────┴────┐
    │         │
 SegFormer  ZoeDepth
 (segments)  (depth)
    │         │
    └────┬────┘
         │
   Chronos Forecast ──► flood_risk_flag / fire_risk_flag
         │
   XGBoost Classifier ──► risk_tier (low/moderate/high/critical)
         │
   RAG Retriever (pgvector + CrossEncoder)
         │
   Streamlit Dashboard
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Satellite imagery | Sentinel-2 via `sentinelsat`, `rasterio` |
| Segmentation | SegFormer (EuroSAT fine-tuned) |
| Depth estimation | ZoeDepth (`Intel/zoedepth-nyu`) |
| Time series | Chronos (`amazon/chronos-t5-small`) |
| Tabular ML | XGBoost 2.0 |
| Embeddings | `all-MiniLM-L6-v2`, CLIP |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | PostgreSQL + pgvector |
| Cache | Redis |
| Pipeline | Prefect 3 |
| Storage | S3-compatible (AWS / MinIO) |
| Dashboard | Streamlit |
| Backend | FastAPI (Phase 3) |
| Infra | Docker Compose |

---

## Quickstart

### Prerequisites

- Docker and Docker Compose
- AWS credentials (or MinIO for local S3)
- Python 3.11+

### 1. Clone and configure

```bash
git clone https://github.com/zimpele/ZimRadar.git
cd ZimRadar
cp .env.example .env
# Edit .env with your API keys (see below)
```

### 2. Required environment variables

```env
DATABASE_URL=postgresql+asyncpg://zimradar:password@localhost:5432/zimradar
REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_TILES=zimradar-tiles
S3_BUCKET_PDFS=zimradar-pdfs
```

### 3. Start services

```bash
docker compose up -d
```

Services:
- Streamlit dashboard → `http://localhost:8501`
- Prefect UI → `http://localhost:4200`
- PostgreSQL → `localhost:5432`
- Redis → `localhost:6379`

### 4. Run the DB migration

```bash
docker compose exec postgres psql -U zimradar -d zimradar \
  -f /path/to/src/storage/migrations/001_initial.sql
docker compose exec postgres psql -U zimradar -d zimradar \
  -f /path/to/src/storage/migrations/002_phase2.sql
```

### 5. Run tests

```bash
pip install uv && uv pip install -e ".[dev]"
pytest tests/ --ignore=tests/evals/ -v
```

Eval tests (slow, require model downloads):
```bash
pytest tests/evals/ -v -m slow
```

---

## Project structure

```
src/
  ingestion/      # Sentinel-2, NOAA, FEMA, OSM fetchers
  pipeline/
    segmentation.py   # SegFormer land-use classification
    depth.py          # ZoeDepth flood zone detection
    forecasting.py    # Chronos 30/60/90-day risk forecast
    classifier.py     # XGBoost risk tier classification
  rag/
    chunking.py       # Token-aware text chunking
    embed.py          # Text and image embedders
    retriever.py      # pgvector + CrossEncoder retrieval
  storage/
    db.py             # Async SQLAlchemy session
    models.py         # ORM models
    cache.py          # Redis inference cache
    s3.py             # S3 tile and model storage
    migrations/       # SQL migration files
  dashboard/
    app.py            # Streamlit dashboard
  config.py           # Pydantic settings

tests/
  pipeline/           # Unit tests for each ML module
  rag/                # Retriever tests
  storage/            # Cache and S3 tests
  evals/              # Regression evals (slow, CI-gated)
  fixtures/           # Synthetic holdout data
```

---

## Evals

CI runs regression evals on every pull request:

| Eval | Metric | Threshold |
|---|---|---|
| Segmentation | mIoU on EuroSAT test split | ≥ 0.75 |
| Chronos CRPS | 90-day precipitation holdout | < 1.60 |
| XGBoost AUC-ROC | 5-fold stratified CV | ≥ 0.80 |

---

## Roadmap

- [x] Phase 1 — Data ingestion, segmentation, vector store, Streamlit MVP
- [x] Phase 2 — ZoeDepth, Chronos forecasting, XGBoost classifier, RAG retriever
- [ ] Phase 3 — LangGraph agent orchestration, report generation, Validator Agent
- [ ] Phase 4 — FastAPI, PDF export, production hardening, public demo

---

