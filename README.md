# ZimRadar

**Multi-agent AI system for county-level climate and infrastructure risk assessment.**

ZimRadar fuses satellite imagery, NOAA weather records, FEMA disaster history, and the FEMA National Risk Index to produce per-county risk scores, trend forecasts, and LLM-generated narrative reports — all orchestrated through a Prefect pipeline and surfaced in a live Streamlit dashboard.

---

## Key metrics

| Metric | Value |
|---|---|
| XGBoost AUC-ROC (5-fold CV) | **0.92** |
| Training samples | 3 324 US counties |
| Features | 12 engineered |
| FEMA NRI counties | 3 232 |
| NOAA stations mapped | 1 571+ counties |
| Risk tiers | low / moderate / high / critical |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
│  Sentinel-2 · NOAA CDO · FEMA Declarations · FEMA NRI      │
│  FEMA NFIP · OSM · Census · USGS 3DEP                      │
└────────────────────┬────────────────────────────────────────┘
                     │  Prefect 3 pipeline
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                          │
│  ingest-fema  ·  ingest-fema-nri  ·  ingest-noaa-counties  │
│  ingest-noaa  ·  ingest-sentinel2                           │
└──────────┬──────────────────────────┬───────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────┐
│  ML PIPELINE     │       │  VECTOR STORE         │
│                  │       │                        │
│  SegFormer       │       │  pgvector (pg16)       │
│  (segmentation)  │       │  FEMA report chunks    │
│                  │       │  Satellite tile embeds │
│  ZoeDepth        │       │  CrossEncoder rerank   │
│  (depth / flood) │       └──────────────────────┘
│                  │
│  Chronos         │       ┌──────────────────────┐
│  (30/60/90d      │       │  FEATURE STORE        │
│   forecast)      │       │                        │
│                  │       │  flood_events_5yr      │
│  XGBoost         │◄──────│  precip_trend (NOAA)  │
│  Classifier      │       │  vegetation_loss_pct   │
│  AUC 0.92        │       │  urban_density         │
│                  │       │  elevation_variance    │
└────────┬─────────┘       │  nri_risk_score        │
         │                 │  nri_eal_score          │
         ▼                 │  nri_flood_risks        │
┌──────────────────┐       │  nri_fire_risks         │
│  RAG RETRIEVER   │       │  nri_heat_risks         │
│  + LangSmith     │       └──────────────────────┘
│  tracing         │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│  DASHBOARD & API             │
│  Streamlit · FastAPI         │
│  Folium county map           │
│  Risk tier overlays          │
└──────────────────────────────┘
```

---

## Feature engineering (XGBoost)

| Feature | Source | Coverage |
|---|---|---|
| `flood_events_5yr` | FEMA Declarations | ~3 300 counties |
| `avg_precipitation_trend` | NOAA CDO (bulk station lookup) | ~1 600 counties |
| `vegetation_loss_pct` | SegFormer / Sentinel-2 | tracked regions |
| `urban_density` | SegFormer / Sentinel-2 | tracked regions |
| `elevation_variance` | ZoeDepth depth results | tracked regions |
| `infrastructure_age_proxy` | static (OSM planned) | — |
| `nri_risk_score` | FEMA National Risk Index | 3 232 counties |
| `nri_eal_score` | FEMA NRI (expected annual loss) | 3 232 counties |
| `nri_sovi_score` | FEMA NRI (social vulnerability) | 3 232 counties |
| `nri_flood_risks` | FEMA NRI (coastal + inland flood) | 3 232 counties |
| `nri_fire_risks` | FEMA NRI (wildfire) | 3 232 counties |
| `nri_heat_risks` | FEMA NRI (heat wave) | 3 232 counties |

---

## What's implemented

### Data ingestion
- **Sentinel-2** — nightly tile fetches via `sentinelsat`; processed tiles stored in S3
- **NOAA CDO** — daily precipitation, temperature, soil moisture per tracked region; bulk county climate summary (2yr trend) across all FEMA counties
- **FEMA Declarations** — full 60k+ record bulk sync with 5-digit county FIPS; incremental delta updates
- **FEMA National Risk Index** — 3 232 US counties, 15 hazard columns, fetched from ArcGIS FeatureServer

### ML pipeline
- **SegFormer** — EuroSAT-fine-tuned land-use segmentation (water / vegetation / urban / bare soil / burn scar)
- **ZoeDepth** — monocular depth estimation → flood-accumulation zone detection
- **Chronos** — 30/60/90-day probabilistic flood and fire risk flags
- **XGBoost classifier** — 4-tier county risk label with balanced sample weights; 5-fold stratified CV; AUC 0.92

### Retrieval-augmented generation
- **pgvector** — FEMA report chunks + satellite tile embeddings (384-dim `all-MiniLM-L6-v2`, 512-dim CLIP)
- **CrossEncoder** — `ms-marco-MiniLM-L-6-v2` reranking with metadata filters
- **LangSmith** — tracing for all LLM and retrieval calls

### Observability
- **Prefect 3** — 6 deployments: `ingest-fema`, `ingest-fema-nri`, `ingest-noaa`, `ingest-noaa-counties`, `ingest-sentinel2`, `train-xgboost-classifier`
- **Streamlit** — live interactive Folium map with county-level risk tier overlays, sidebar county picker

---

## Tech stack

| Layer | Technology |
|---|---|
| Satellite imagery | Sentinel-2 via `sentinelsat`, `rasterio` |
| Segmentation | SegFormer (EuroSAT fine-tuned) |
| Depth estimation | ZoeDepth (`Intel/zoedepth-nyu`) |
| Time series | Chronos (`amazon/chronos-t5-small`) |
| Tabular ML | XGBoost 2.0 + scikit-learn |
| Embeddings | `all-MiniLM-L6-v2`, CLIP |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | PostgreSQL 16 + pgvector |
| Cache | Redis 7 |
| Pipeline | Prefect 3 |
| Tracing | LangSmith |
| Storage | S3-compatible (AWS / MinIO) |
| Dashboard | Streamlit + Folium |
| Backend | FastAPI |
| Infra | Docker Compose |

---

## Quickstart

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- AWS credentials (or MinIO for local S3)
- NOAA CDO API key — free at [ncdc.noaa.gov](https://www.ncdc.noaa.gov/cdo-web/token)

### 1. Clone and configure

```bash
git clone https://github.com/zimpele/ZimRadar.git
cd ZimRadar
cp .env.example .env
# Fill in your credentials — see .env.example for all required vars
```

### 2. Required environment variables

```env
DATABASE_URL=postgresql+asyncpg://zimradar:password@localhost:5432/zimradar
REDIS_URL=redis://localhost:6379
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_TILES=zimradar-tiles
S3_BUCKET_PDFS=zimradar-pdfs
NOAA_API_KEY=...
OPENROUTER_API_KEY=...      # LLM calls via OpenRouter
LANGSMITH_API_KEY=...       # optional — enables LangSmith tracing
```

### 3. Start services

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| Prefect UI | http://localhost:4200 |
| FastAPI | http://localhost:8000 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

### 4. Apply migrations

```bash
for f in src/storage/migrations/*.sql; do
  docker compose exec -T postgres psql -U zimradar -d zimradar < "$f"
done
```

### 5. Seed data (run in Prefect UI or CLI)

Run these flows in order:

```
ingest-fema              # FEMA disaster declarations (~60k records)
ingest-fema-nri          # FEMA National Risk Index (3 232 counties)
ingest-noaa-counties     # NOAA station lookup + 2yr precipitation trend
train-xgboost-classifier # Train and save model to S3
```

### 6. Run tests

```bash
pip install uv && uv pip install -e ".[dev]"
pytest tests/ --ignore=tests/evals/ -v
```

---

## Project structure

```
src/
  ingestion/
    fema.py              # FEMA disaster declarations (60k+ records)
    fema_nri.py          # FEMA National Risk Index (ArcGIS FeatureServer)
    noaa.py              # NOAA daily observations (per tracked region)
    noaa_counties.py     # Bulk NOAA station lookup (all FEMA counties)
    sentinel2.py         # Sentinel-2 tile fetch + processing
    geo_admin.py         # GeoAdmin region helpers
    osm.py               # OSM infrastructure data
  pipeline/
    train_flow.py        # Prefect training flow (build features → CV → save)
    classifier.py        # XGBoost feature engineering + inference
    segmentation.py      # SegFormer land-use classification
    depth.py             # ZoeDepth flood zone detection
    forecasting.py       # Chronos 30/60/90d risk forecast
  rag/
    chunking.py          # Token-aware text chunking
    embed.py             # Text and image embedders
    retriever.py         # pgvector + CrossEncoder retrieval
  storage/
    db.py                # Async SQLAlchemy session factory
    models.py            # ORM models
    cache.py             # Redis inference cache
    s3.py                # S3 tile and model storage
    migrations/          # Versioned SQL migrations (001–007)
  dashboard/
    app.py               # Streamlit + Folium interactive map
  api/
    main.py              # FastAPI app with LangSmith lifespan
  config.py              # Pydantic settings (env-driven)

tests/
  pipeline/              # Unit tests for each ML module
  rag/                   # Retriever and chunking tests
  storage/               # Cache and S3 tests
  evals/                 # Regression evals (slow, CI-gated)
  fixtures/              # Synthetic holdout data
```

---

## Eval thresholds (CI)

| Eval | Metric | Threshold | Current |
|---|---|---|---|
| Segmentation | mIoU on EuroSAT test split | ≥ 0.75 | — |
| Chronos CRPS | 90-day precipitation holdout | < 1.60 | — |
| XGBoost AUC-ROC | 5-fold stratified CV | ≥ 0.80 | **0.92** |

---

## Roadmap

- [x] Phase 1 — Sentinel-2 ingestion, segmentation, pgvector, Streamlit MVP
- [x] Phase 2 — ZoeDepth, Chronos forecasting, XGBoost classifier (AUC 0.92), RAG retriever
- [x] Phase 2.5 — FEMA NRI integration, bulk NOAA county coverage, LangSmith tracing
- [ ] Phase 3 — LangGraph agent orchestration, LLM narrative reports, Validator Agent
- [ ] Phase 4 — FastAPI async `/assess`, PDF export, public demo
