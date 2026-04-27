# ZimRadar — Full System Design

**Date:** 2026-04-27  
**Status:** Approved  
**Scope:** Full system spec — all 6 layers, AWS deployment, CI/CD

---

## Overview

ZimRadar is an open-source, multi-agent AI system that ingests satellite imagery, weather data, and historical disaster records to generate per-region climate risk reports. A user enters a US county name or coordinates; the system returns a structured risk report with segmented satellite overlays, trend forecasts, historical context from FEMA records, and an LLM-generated narrative with inline citations.

**Primary goal:** Portfolio project demonstrating production AI engineering skills across multi-agent orchestration, RAG pipelines, computer vision, time series forecasting, and LLMOps.

**Deliverable:** Public GitHub repo + live deployed demo (AWS) + Docker Compose for local dev.

---

## Architecture Approach

Full stack as designed — all 6 layers:

1. Data ingestion (Prefect)
2. ML processing pipeline (HuggingFace models)
3. Storage & retrieval (PostgreSQL + pgvector + LlamaIndex)
4. Multi-agent orchestration (LangGraph)
5. Evaluation harness (CI-gated)
6. Output (Streamlit MVP → Next.js + FastAPI + PDF)

---

## Layer 1 — Data Ingestion

Four independent Prefect flows:

| Flow | Schedule | Source | Output |
|---|---|---|---|
| `ingest_sentinel2` | Nightly | sentinelsat → SciHub | Raw `.SAFE` tiles → S3 |
| `ingest_noaa` | Nightly | NOAA CDO REST API | Daily weather rows → Postgres |
| `ingest_fema` | Weekly | OpenFEMA REST API | Disaster declarations → Postgres (delta via `lastRefresh`) |
| `ingest_osm` | Monthly | Overpass API | Building/road GeoJSON → Postgres |

**Error handling:** 3-attempt retry with exponential backoff per flow. Failed runs log to a `failed_ingestion` table (region, flow name, timestamp, error message). All writes are idempotent (upsert on natural keys).

**Sentinel-2 storage:** Raw `.SAFE` tiles stored in `s3://zimradar-tiles/sentinel2/{region_id}/{date}/`. A downstream Prefect task preprocesses tiles (crop to 256×256, normalize bands) and writes processed tiles back to S3. Raw tiles are never deleted — re-running the ML pipeline does not require re-downloading.

**Region management:** Tracked regions are rows in a `regions` table (bounding box, name, active flag). Adding a new region requires only a database insert.

**Data access (all free):**
- Copernicus / Sentinel-2: register at scihub.copernicus.eu → API key
- NOAA CDO: free API key at ncdc.noaa.gov
- OpenFEMA: no auth required
- OpenStreetMap / Overpass: no auth required

---

## Layer 2 — ML Processing Pipeline

Four HuggingFace models, each wrapped as an independent Prefect task. All run on CPU via the MPS backend (Apple Silicon) or standard CPU elsewhere.

### Image Segmentation
- **Model:** SegFormer-B2 fine-tuned on EuroSAT (HuggingFace Hub)
- **Input:** 256×256 RGB tiles from S3
- **Output:** Per-pixel land-use mask (water / vegetation / urban / bare_soil / burn_scar)
- **Storage:** GeoJSON polygons with area stats per class → `segmentation_results` table

### Depth Estimation
- **Model:** ZoeDepth (zero-shot, no fine-tuning)
- **Input:** Same 256×256 tiles
- **Output:** Relative depth map → lowest-10% elevation pixels flagged as flood-accumulation zones
- **Storage:** Flood zone GeoJSON alongside segmentation results
- **Note:** Used as a feature in the risk classifier; not served directly to users

### Time Series Forecasting
- **Model:** Chronos (Amazon, HuggingFace Hub)
- **Input:** 2-year daily NOAA weather + NDVI (computed from segmentation vegetation mask area over time)
- **Output:** 30/60/90-day forecast distributions → `forecasts` table
- **Risk flags (computed at write time):**
  - `flood_risk_flag`: P(precipitation > 95th percentile for 3+ consecutive days) > 0.3
  - `fire_risk_flag`: P(temperature > 40°C for 7+ days AND NDVI declining) > 0.3

### Tabular Classification
- **Model:** XGBoost trained offline on engineered features
- **Features:** `flood_events_5yr`, `avg_precipitation_trend`, `vegetation_loss_pct`, `urban_density`, `elevation_variance`, `infrastructure_age_proxy`
- **Output:** Risk tier (low / moderate / high / critical) + confidence score → `risk_assessments` table
- **Artifact:** Model stored in S3, loaded at inference time
- **Training data:** FEMA-derived labels with stratified k-fold cross-validation

---

## Layer 3 — Storage & Retrieval

Single PostgreSQL 16 instance with pgvector extension. Redis for inference caching.

### Schema (key tables)

| Table | Contents |
|---|---|
| `regions` | Bounding boxes, names, active flags |
| `sentinel2_tiles` | Tile metadata, S3 paths, ingestion timestamps |
| `noaa_observations` | Daily weather rows per station |
| `fema_declarations` | Disaster records, county FIPS, type, dates |
| `segmentation_results` | GeoJSON masks, area stats, tile FK |
| `risk_assessments` | Risk tier, confidence, composite score, region FK |
| `forecasts` | Chronos output distributions, risk flags, region FK |
| `reports` | Generated narrative reports, factuality score, citations, retry count |
| `image_embeddings` | CLIP embeddings (pgvector column), tile FK, metadata |
| `text_embeddings` | MiniLM embeddings (pgvector column), source doc FK, chunk text |
| `failed_ingestion` | Error log for failed Prefect flow runs |

### RAG Pipeline (LlamaIndex)

Hybrid retrieval over `text_embeddings` (FEMA reports + NOAA station metadata):

1. Semantic search via pgvector cosine similarity
2. Metadata filters: date range, county FIPS, disaster type
3. Cross-encoder re-ranking (`ms-marco-MiniLM-L-6-v2`, HuggingFace)
4. Top-5 chunks returned with source citations to Report Agent

**Chunking:** FEMA reports chunked at 512 tokens with 64-token overlap, embedded with `all-MiniLM-L6-v2`. Embedding happens at ingestion time inside `ingest_fema`.

### Redis Cache

Inference results cached by `(tile_s3_path, model_version)` key. TTL: 7 days. Prevents re-running SegFormer/ZoeDepth on unchanged tiles.

---

## Layer 4 — Multi-Agent Orchestration (LangGraph)

Four agents in a directed graph with a conditional re-generation loop.

```
Ingest Agent → Analysis Agent → Report Agent → Validator Agent
                                      ↑                |
                                      └────────────────┘
                                   (if factuality < 0.8, max 2 retries)
```

### Agents

**Ingest Agent**
- Receives region query, triggers Prefect data fetches for the region, runs CV pipeline (segmentation + depth estimation) on fresh tiles, generates CLIP embeddings for image retrieval, writes all results to Postgres.
- Tools: `sentinelsat`, `rasterio`, SegFormer, ZoeDepth, CLIP, S3 read/write

**Analysis Agent**
- Pulls NOAA observations + segmentation results from Postgres, runs Chronos forecast, runs XGBoost classifier, computes composite risk score: `score = W1 * xgboost_confidence + W2 * flood_risk_flag + W3 * fire_risk_flag` where `W1=0.6, W2=0.2, W3=0.2` by default (configurable via `RISK_SCORE_WEIGHTS` env var as a comma-separated triple).
- Tools: Chronos, XGBoost, scoring function, Postgres read/write

**Report Agent**
- Queries LlamaIndex RAG pipeline with the region + risk data as context, builds structured prompt with retrieved FEMA/NOAA passages, calls Ollama (`gemma2:9b` default, configurable via `OLLAMA_MODEL` env var), generates narrative report with inline citations in `[n]` format.
- Tools: LlamaIndex retriever, Ollama, citation formatter

**Validator Agent**
- Runs extractive QA (`deepset/roberta-base-squad2`, CPU) to verify each cited claim can be grounded in a retrieved source passage. Scores factuality 0–1. If score ≥ 0.8 → done. If score < 0.8 and retry count < 2 → routes back to Report Agent with feedback. If retries exhausted → saves report with `low_confidence = true` flag.
- Tools: `deepset/roberta-base-squad2`, Ollama (LLM-as-judge for overall factuality score)

### Shared State (TypedDict)

```python
class ZimRadarState(TypedDict):
    region_query: str
    tile_paths: list[str]
    segmentation_results: dict
    depth_map: dict
    forecast: dict
    risk_tier: str
    risk_score: float
    retrieved_context: list[dict]
    report_draft: str
    factuality_score: float
    retry_count: int
    final_report: str | None
    low_confidence: bool
```

### Local LLM

- **Default model:** `gemma2:9b` (Ollama) — 5.5 GB, runs comfortably on M1 Pro 16 GB with room for the full stack
- **Configuration:** `OLLAMA_MODEL` env var — swap to `qwen2.5:14b` or any Ollama-supported model
- **Endpoint:** `http://ollama:11434` (Docker service name)

---

## Layer 5 — Evaluation Harness

All evals run via `make eval` locally and as a GitHub Actions CI step on every PR. Any regression blocks merge.

| Dimension | Method | Metric | Regression threshold |
|---|---|---|---|
| Segmentation accuracy | EuroSAT test split | mIoU | < 0.82 blocks merge |
| Forecast calibration | 2-year NOAA holdout backtest | CRPS | > 0.90 blocks merge |
| Tabular classification | Stratified 5-fold on FEMA labels | AUC-ROC | < 0.80 blocks merge |
| Report factuality | Validator Agent on 50 fixed test reports | Mean factuality score | < 0.78 blocks merge |
| Citation accuracy | Extractive QA on known source→claim pairs | F1 | < 0.85 blocks merge |
| E2E latency | LangSmith trace on 10 fixed regions | p95 | > 60s blocks merge |

**Test fixtures:** Stored in `tests/fixtures/`. The 50 test reports and 10 E2E regions are checked in as JSON. The NOAA holdout slice is a 500-row CSV. EuroSAT test split is pulled from HuggingFace at eval time (cached after first run).

**LangSmith:** Every agent node emits a trace. Dashboard shows per-agent latency, Ollama token counts, factuality trend over time, and retry rate.

---

## Layer 6 — Output

### Dashboard Layout

Map + sidebar panel: interactive map on the left (~60% width), report panel on the right (~40%). Clicking a region on the map loads its report in the sidebar. Both Streamlit MVP and Next.js use this same layout.

**Streamlit MVP (Phase 1)**
- `leafmap` for the interactive map with GeoJSON risk overlays coloured by tier
- Sidebar: narrative report, risk stat cards (score, flood probability, FEMA declaration count), Chronos forecast chart
- "Run Assessment" button calls the LangGraph pipeline **directly as a Python function** (no FastAPI dependency in Phase 1) and streams progress via `st.status`. The FastAPI layer wraps the same pipeline in Phase 3.

**Next.js + Mapbox GL (Phase 4)**
- Mapbox GL for vector tile map with custom risk-tier colour ramps
- React components for report panel, stat cards, time series chart
- Next.js App Router with server components for initial data fetch
- Replaces Streamlit as the live demo URL; Streamlit remains available locally

### FastAPI REST API

Base URL: `https://api.zimradar.io` (production) / `http://localhost:8000` (local)

| Endpoint | Method | Description |
|---|---|---|
| `/assess` | POST | `{region: string, date_range: [start, end]}` → triggers LangGraph pipeline → returns `{report_id}` |
| `/report/{report_id}` | GET | Full report JSON: risk score, tier, narrative, citations, forecast, segmentation stats |
| `/report/{report_id}/pdf` | GET | PDF export via weasyprint; cached in S3 after first generation |
| `/regions` | GET | List tracked regions with latest risk scores |

Auth: `Authorization: Bearer <api_key>` header, validated as FastAPI middleware. All endpoints async.

### PDF Export

`GET /report/{id}/pdf` generates a PDF via `weasyprint` from an HTML template. Includes map screenshot (headless Playwright), risk stats, full narrative, and citations. Stored in `s3://zimradar-pdfs/{report_id}.pdf` on first generation; subsequent requests served from S3.

---

## Infrastructure

### Docker Compose (local dev)

```
postgres    pgvector/pgvector:pg16       persistent volume
redis       redis:7-alpine               inference cache
ollama      ollama/ollama                local model cache mount
prefect     prefectHQ/prefect:3-latest   flow scheduler + UI :4200
api         local build                  FastAPI :8000
streamlit   local build                  Streamlit :8501
worker      local build                  Prefect worker + LangGraph runner
```

- `make up` — start full stack
- `make eval` — run eval suite
- `make pull-model` — pull `gemma2:9b` into Ollama on first setup
- `.env.example` checked into repo; actual `.env` gitignored

### AWS Deployment (Terraform-managed)

| Resource | Type | Notes |
|---|---|---|
| EC2 | `t3.xlarge` spot instance | Runs full Docker Compose stack |
| RDS PostgreSQL | `db.t3.medium` | pgvector extension, replaces local Postgres |
| ElastiCache Redis | `cache.t3.micro` | Replaces local Redis |
| S3 | Two buckets: `zimradar-tiles`, `zimradar-pdfs` | Tile storage + PDF cache |
| ALB | Application Load Balancer | HTTPS termination → FastAPI + Next.js |
| ECR | Elastic Container Registry | Built Docker images |

Terraform state stored in S3 with DynamoDB locking. `terraform apply` from `infra/` provisions everything from scratch.

### CI/CD (GitHub Actions)

1. **On PR:** run eval suite → lint → type-check → block merge on any failure
2. **On merge to `main`:** build Docker images → push to ECR → SSH deploy to EC2 → smoke test live URL

---

## Repo Structure

```
zimradar/
├── infra/                  # Terraform (AWS)
├── docker-compose.yml
├── .env.example
├── Makefile
├── src/
│   ├── ingestion/          # Prefect flows (sentinel2, noaa, fema, osm)
│   ├── pipeline/           # ML models (segformer, zoedepth, chronos, xgboost)
│   ├── storage/            # Postgres schema, pgvector queries, Redis cache
│   ├── rag/                # LlamaIndex retriever, chunking, embedding
│   ├── agents/             # LangGraph graph + 4 agent nodes
│   ├── api/                # FastAPI app
│   ├── dashboard/          # Streamlit MVP
│   └── frontend/           # Next.js app
├── tests/
│   ├── fixtures/           # Test reports, E2E regions, NOAA holdout
│   └── evals/              # Eval suite (segmentation, forecast, tabular, factuality)
└── docs/
    └── superpowers/specs/
```

---

## Implementation Roadmap

### Phase 1 — Foundation (weeks 1–3)
- Repo structure, Docker Compose, CI skeleton
- Sentinel-2 + NOAA + FEMA ingestion flows
- SegFormer segmentation pipeline with EuroSAT eval
- pgvector schema, FEMA report indexing
- Streamlit MVP (map + basic report view)
- **Deliverable:** Data pipeline runs end-to-end, segmentation eval passes

### Phase 2 — Intelligence (weeks 4–6)
- ZoeDepth terrain analysis
- Chronos forecasting pipeline with NOAA backtest
- XGBoost tabular classifier training
- LlamaIndex RAG pipeline with hybrid retrieval + re-ranking
- **Deliverable:** All models running, individual eval metrics baselined

### Phase 3 — Orchestration (weeks 7–9)
- LangGraph 4-agent graph
- Validator Agent with extractive QA + re-generation loop
- LangSmith tracing across all agents
- FastAPI REST endpoints
- **Deliverable:** Full `query → report` pipeline working, traces visible

### Phase 4 — Polish & Ship (weeks 10–12)
- Next.js + Mapbox GL dashboard
- PDF export
- AWS deployment (Terraform + CI/CD)
- CI eval regression checks
- README, architecture docs, demo GIF
- **Deliverable:** Live demo URL, open-sourced repo, portfolio-ready

---

## Key Configuration (env vars)

| Var | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `gemma2:9b` | Ollama model for report generation + LLM judge |
| `OLLAMA_URL` | `http://ollama:11434` | Ollama service endpoint |
| `DATABASE_URL` | `postgresql://zimradar:password@postgres:5432/zimradar` | Postgres connection string |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string |
| `S3_BUCKET_TILES` | `zimradar-tiles` | S3 bucket for Sentinel-2 tiles |
| `S3_BUCKET_PDFS` | `zimradar-pdfs` | S3 bucket for PDF exports |
| `SENTINELSAT_USER` | — | Copernicus SciHub username |
| `SENTINELSAT_PASS` | — | Copernicus SciHub password |
| `NOAA_API_KEY` | — | NOAA CDO API key |
| `LANGSMITH_API_KEY` | — | LangSmith tracing |
| `API_KEY` | — | ZimRadar REST API auth key |
| `RISK_SCORE_WEIGHTS` | `0.6,0.2,0.2` | Weights for composite score: XGBoost confidence, flood flag, fire flag |
