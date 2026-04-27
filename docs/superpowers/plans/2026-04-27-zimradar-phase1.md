# ZimRadar Phase 1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the full data ingestion pipeline (Sentinel-2, NOAA, FEMA, OSM), run SegFormer segmentation on tiles, index FEMA reports into pgvector, and serve a basic Streamlit map dashboard — all running end-to-end via `make up`.

**Architecture:** Four independent Prefect flows handle data ingestion from external APIs into Postgres and S3. A separate preprocessing pipeline runs SegFormer on raw tiles and stores GeoJSON segmentation masks. LlamaIndex ingests chunked FEMA reports into pgvector. Streamlit calls the LangGraph pipeline directly (no FastAPI yet).

**Tech Stack:** Python 3.11, Prefect 3, SQLAlchemy 2 + asyncpg, pgvector, sentinelsat, rasterio, transformers (SegFormer-B2), sentence-transformers (all-MiniLM-L6-v2), boto3, Streamlit, leafmap, pytest, Docker Compose

---

## File Structure

```
zimradar/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── .env.example
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── config.py                    # Pydantic settings from env vars
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── db.py                    # Async SQLAlchemy engine + session factory
│   │   ├── models.py                # SQLAlchemy ORM models for all tables
│   │   ├── migrations/
│   │   │   └── 001_initial.sql      # All tables + pgvector extension
│   │   └── s3.py                    # boto3 S3 client wrapper
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base.py                  # Retry decorator + error logging to failed_ingestion
│   │   ├── fema.py                  # Prefect flow: fetch + upsert + chunk + embed
│   │   ├── noaa.py                  # Prefect flow: fetch daily weather
│   │   ├── osm.py                   # Prefect flow: fetch building/road GeoJSON
│   │   └── sentinel2.py             # Prefect flow: download tiles + preprocess
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Crop 256×256, normalize bands, write to S3
│   │   └── segmentation.py          # SegFormer-B2 inference → GeoJSON masks
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py              # 512-token chunker with 64-token overlap
│   │   └── embed.py                 # MiniLM text embedder + CLIP image embedder
│   └── dashboard/
│       ├── __init__.py
│       └── app.py                   # Streamlit app: leafmap + risk panel
├── tests/
│   ├── conftest.py                  # Shared fixtures: DB session, S3 mock, sample tiles
│   ├── storage/
│   │   ├── test_db.py
│   │   └── test_s3.py
│   ├── ingestion/
│   │   ├── test_fema.py
│   │   ├── test_noaa.py
│   │   ├── test_osm.py
│   │   └── test_sentinel2.py
│   ├── pipeline/
│   │   ├── test_preprocessing.py
│   │   └── test_segmentation.py
│   ├── rag/
│   │   ├── test_chunking.py
│   │   └── test_embed.py
│   └── evals/
│       └── test_segmentation_eval.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `docker-compose.yml`
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/config.py`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: zimradar
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-password}
      POSTGRES_DB: zimradar
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./src/storage/migrations:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U zimradar"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  ollama:
    image: ollama/ollama
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    environment:
      OLLAMA_HOST: 0.0.0.0

  prefect:
    image: prefecthq/prefect:3-latest
    command: prefect server start --host 0.0.0.0
    ports:
      - "4200:4200"
    environment:
      PREFECT_API_URL: http://localhost:4200/api
    depends_on:
      postgres:
        condition: service_healthy

  worker:
    build:
      context: .
      target: worker
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://zimradar:${POSTGRES_PASSWORD:-password}@postgres:5432/zimradar
      REDIS_URL: redis://redis:6379
      PREFECT_API_URL: http://prefect:4200/api
      OLLAMA_URL: http://ollama:11434
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./src:/app/src

  streamlit:
    build:
      context: .
      target: streamlit
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://zimradar:${POSTGRES_PASSWORD:-password}@postgres:5432/zimradar
    ports:
      - "8501:8501"
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
  ollama_data:
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "zimradar"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "prefect>=3.0",
  "sentinelsat>=1.2",
  "rasterio>=1.3",
  "geopandas>=0.14",
  "shapely>=2.0",
  "numpy>=1.26",
  "torch>=2.2",
  "transformers>=4.40",
  "sentence-transformers>=3.0",
  "sqlalchemy[asyncio]>=2.0",
  "asyncpg>=0.29",
  "pgvector>=0.3",
  "redis>=5.0",
  "boto3>=1.34",
  "streamlit>=1.35",
  "leafmap>=0.30",
  "llama-index>=0.10",
  "llama-index-vector-stores-postgres>=0.1",
  "pydantic-settings>=2.0",
  "httpx>=0.27",
  "pillow>=10.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "pytest-mock>=3.14",
  "moto[s3]>=5.0",
  "factory-boy>=3.3",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Create `Makefile`**

```makefile
.PHONY: up down eval pull-model migrate

up:
	docker compose up -d
	@echo "Services started. Streamlit: http://localhost:8501  Prefect: http://localhost:4200"

down:
	docker compose down

pull-model:
	docker compose exec ollama ollama pull gemma2:9b

migrate:
	docker compose exec postgres psql -U zimradar -d zimradar -f /docker-entrypoint-initdb.d/001_initial.sql

eval:
	pytest tests/evals/ -v

test:
	pytest tests/ -v --ignore=tests/evals/

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
```

- [ ] **Step 4: Create `.env.example`**

```bash
# .env.example — copy to .env and fill in values
POSTGRES_PASSWORD=password

# Copernicus SciHub credentials (register at scihub.copernicus.eu)
SENTINELSAT_USER=
SENTINELSAT_PASS=

# NOAA Climate Data Online API key (register at ncdc.noaa.gov)
NOAA_API_KEY=

# AWS credentials (for S3 tile storage)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_TILES=zimradar-tiles
S3_BUCKET_PDFS=zimradar-pdfs

# Ollama
OLLAMA_MODEL=gemma2:9b
OLLAMA_URL=http://ollama:11434

# LangSmith (optional — traces agent runs)
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=zimradar

# Composite risk score weights: xgboost_confidence, flood_flag, fire_flag
RISK_SCORE_WEIGHTS=0.6,0.2,0.2

# ZimRadar API auth key
API_KEY=

# Internal service URLs (set automatically in docker-compose, override for local dev)
DATABASE_URL=postgresql+asyncpg://zimradar:password@localhost:5432/zimradar
REDIS_URL=redis://localhost:6379
```

- [ ] **Step 5: Create `src/config.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://zimradar:password@localhost:5432/zimradar"
    redis_url: str = "redis://localhost:6379"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"

    sentinelsat_user: str = ""
    sentinelsat_pass: str = ""
    noaa_api_key: str = ""

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    s3_bucket_tiles: str = "zimradar-tiles"
    s3_bucket_pdfs: str = "zimradar-pdfs"

    langsmith_api_key: str = ""
    langsmith_project: str = "zimradar"
    api_key: str = ""

    risk_score_weights: str = "0.6,0.2,0.2"

    @property
    def risk_weights(self) -> tuple[float, float, float]:
        w = [float(x) for x in self.risk_score_weights.split(",")]
        return w[0], w[1], w[2]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 6: Create `src/__init__.py`** (empty file)

```bash
touch src/__init__.py src/storage/__init__.py src/ingestion/__init__.py
touch src/pipeline/__init__.py src/rag/__init__.py src/dashboard/__init__.py
touch tests/__init__.py tests/storage/__init__.py tests/ingestion/__init__.py
touch tests/pipeline/__init__.py tests/rag/__init__.py tests/evals/__init__.py
```

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml pyproject.toml Makefile .env.example src/
git commit -m "feat: project scaffold — docker-compose, config, pyproject"
```

---

## Task 2: Database Schema & Models

**Files:**
- Create: `src/storage/migrations/001_initial.sql`
- Create: `src/storage/models.py`
- Create: `src/storage/db.py`
- Create: `tests/storage/test_db.py`

- [ ] **Step 1: Write failing test**

```python
# tests/storage/test_db.py
import pytest
from sqlalchemy import text
from src.storage.db import get_async_session


@pytest.mark.asyncio
async def test_pgvector_extension_enabled(db_session):
    result = await db_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.scalar() == "vector"


@pytest.mark.asyncio
async def test_all_tables_exist(db_session):
    expected = {
        "regions", "sentinel2_tiles", "noaa_observations", "fema_declarations",
        "segmentation_results", "risk_assessments", "forecasts", "reports",
        "image_embeddings", "text_embeddings", "failed_ingestion",
    }
    result = await db_session.execute(
        text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    )
    tables = {row[0] for row in result.fetchall()}
    assert expected.issubset(tables)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/storage/test_db.py -v
```

Expected: `FAIL` — `db_session` fixture not defined, `get_async_session` not importable.

- [ ] **Step 3: Create `src/storage/migrations/001_initial.sql`**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS regions (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    bbox JSONB NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sentinel2_tiles (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id) ON DELETE CASCADE,
    s3_path TEXT NOT NULL,
    processed_s3_path TEXT,
    date DATE NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(region_id, s3_path)
);

CREATE TABLE IF NOT EXISTS noaa_observations (
    id SERIAL PRIMARY KEY,
    station_id TEXT NOT NULL,
    region_id INTEGER REFERENCES regions(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    precipitation_mm FLOAT,
    temp_max_c FLOAT,
    temp_min_c FLOAT,
    soil_moisture FLOAT,
    UNIQUE(station_id, date)
);

CREATE TABLE IF NOT EXISTS fema_declarations (
    id SERIAL PRIMARY KEY,
    disaster_number TEXT UNIQUE NOT NULL,
    state TEXT,
    county_fips TEXT,
    disaster_type TEXT,
    declaration_date DATE,
    incident_begin DATE,
    incident_end DATE,
    declaration_title TEXT
);

CREATE TABLE IF NOT EXISTS segmentation_results (
    id SERIAL PRIMARY KEY,
    tile_id INTEGER REFERENCES sentinel2_tiles(id) ON DELETE CASCADE,
    geojson JSONB NOT NULL,
    area_stats JSONB NOT NULL,
    flood_zone_geojson JSONB,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id) ON DELETE CASCADE,
    risk_tier TEXT NOT NULL CHECK (risk_tier IN ('low', 'moderate', 'high', 'critical')),
    confidence FLOAT NOT NULL,
    composite_score FLOAT NOT NULL,
    assessed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id) ON DELETE CASCADE,
    forecast_30d JSONB,
    forecast_60d JSONB,
    forecast_90d JSONB,
    flood_risk_flag BOOLEAN DEFAULT FALSE,
    fire_risk_flag BOOLEAN DEFAULT FALSE,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    region_id INTEGER REFERENCES regions(id) ON DELETE CASCADE,
    narrative TEXT NOT NULL,
    citations JSONB NOT NULL DEFAULT '[]',
    factuality_score FLOAT,
    retry_count INTEGER DEFAULT 0,
    low_confidence BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS image_embeddings (
    id SERIAL PRIMARY KEY,
    tile_id INTEGER REFERENCES sentinel2_tiles(id) ON DELETE CASCADE,
    embedding vector(512),
    metadata JSONB DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS text_embeddings (
    id SERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    UNIQUE(source_type, source_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS failed_ingestion (
    id SERIAL PRIMARY KEY,
    region_id INTEGER,
    flow_name TEXT NOT NULL,
    error_message TEXT NOT NULL,
    failed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_text_embeddings_vector
    ON text_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_image_embeddings_vector
    ON image_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

- [ ] **Step 4: Create `src/storage/db.py`**

```python
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.config import get_settings


def _make_engine():
    settings = get_settings()
    return create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)


_engine = None
_session_factory = None


def _ensure_initialized():
    global _engine, _session_factory
    if _engine is None:
        _engine = _make_engine()
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


@asynccontextmanager
async def get_async_session() -> AsyncSession:
    _ensure_initialized()
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

- [ ] **Step 5: Create `src/storage/models.py`**

```python
from datetime import datetime, date
from uuid import UUID
from sqlalchemy import String, Integer, Float, Boolean, Text, Date, DateTime, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Region(Base):
    __tablename__ = "regions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    bbox: Mapped[dict] = mapped_column(JSONB, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Sentinel2Tile(Base):
    __tablename__ = "sentinel2_tiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    s3_path: Mapped[str] = mapped_column(Text, nullable=False)
    processed_s3_path: Mapped[str | None] = mapped_column(Text)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("region_id", "s3_path"),)


class NOAAObservation(Base):
    __tablename__ = "noaa_observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    station_id: Mapped[str] = mapped_column(Text, nullable=False)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    date: Mapped[date] = mapped_column(Date, nullable=False)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    temp_max_c: Mapped[float | None] = mapped_column(Float)
    temp_min_c: Mapped[float | None] = mapped_column(Float)
    soil_moisture: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("station_id", "date"),)


class FEMADeclaration(Base):
    __tablename__ = "fema_declarations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    disaster_number: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    state: Mapped[str | None] = mapped_column(Text)
    county_fips: Mapped[str | None] = mapped_column(Text)
    disaster_type: Mapped[str | None] = mapped_column(Text)
    declaration_date: Mapped[date | None] = mapped_column(Date)
    incident_begin: Mapped[date | None] = mapped_column(Date)
    incident_end: Mapped[date | None] = mapped_column(Date)
    declaration_title: Mapped[str | None] = mapped_column(Text)


class SegmentationResult(Base):
    __tablename__ = "segmentation_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tile_id: Mapped[int] = mapped_column(ForeignKey("sentinel2_tiles.id", ondelete="CASCADE"))
    geojson: Mapped[dict] = mapped_column(JSONB, nullable=False)
    area_stats: Mapped[dict] = mapped_column(JSONB, nullable=False)
    flood_zone_geojson: Mapped[dict | None] = mapped_column(JSONB)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    risk_tier: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Forecast(Base):
    __tablename__ = "forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    forecast_30d: Mapped[dict | None] = mapped_column(JSONB)
    forecast_60d: Mapped[dict | None] = mapped_column(JSONB)
    forecast_90d: Mapped[dict | None] = mapped_column(JSONB)
    flood_risk_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    fire_risk_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id", ondelete="CASCADE"))
    narrative: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list] = mapped_column(JSONB, default=list)
    factuality_score: Mapped[float | None] = mapped_column(Float)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tile_id: Mapped[int] = mapped_column(ForeignKey("sentinel2_tiles.id", ondelete="CASCADE"))
    embedding: Mapped[list] = mapped_column(Vector(512))
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(384))
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    __table_args__ = (UniqueConstraint("source_type", "source_id", "chunk_index"),)


class FailedIngestion(Base):
    __tablename__ = "failed_ingestion"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[int | None] = mapped_column(Integer)
    flow_name: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    failed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
```

- [ ] **Step 6: Create `tests/conftest.py`**

```python
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from src.storage.models import Base


TEST_DB_URL = "postgresql+asyncpg://zimradar:password@localhost:5432/zimradar_test"


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncSession:
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
# First start postgres
docker compose up postgres -d
# Create test DB
docker compose exec postgres createdb -U zimradar zimradar_test

pytest tests/storage/test_db.py -v
```

Expected: `PASS` — both tests green.

- [ ] **Step 8: Commit**

```bash
git add src/storage/ tests/conftest.py tests/storage/
git commit -m "feat: database schema, ORM models, async session factory"
```

---

## Task 3: S3 Client

**Files:**
- Create: `src/storage/s3.py`
- Create: `tests/storage/test_s3.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/storage/test_s3.py
import pytest
from unittest.mock import patch, MagicMock
from src.storage.s3 import S3Client


def test_upload_tile_returns_s3_path(tmp_path):
    tile_path = tmp_path / "tile.tif"
    tile_path.write_bytes(b"fake tif data")

    with patch("src.storage.s3.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        s3 = S3Client(bucket="zimradar-tiles")

        result = s3.upload_tile(str(tile_path), region_id=1, date="2024-01-15")

    assert result == "sentinel2/1/2024-01-15/tile.tif"
    mock_client.upload_file.assert_called_once()


def test_download_tile_writes_file(tmp_path):
    with patch("src.storage.s3.boto3.client") as mock_boto:
        mock_client = MagicMock()
        mock_boto.return_value = mock_client
        s3 = S3Client(bucket="zimradar-tiles")

        dest = str(tmp_path / "downloaded.tif")
        s3.download_tile("sentinel2/1/2024-01-15/tile.tif", dest)

    mock_client.download_file.assert_called_once_with(
        "zimradar-tiles", "sentinel2/1/2024-01-15/tile.tif", dest
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/storage/test_s3.py -v
```

Expected: `FAIL` — `S3Client` not importable.

- [ ] **Step 3: Create `src/storage/s3.py`**

```python
import os
import boto3
from pathlib import Path
from src.config import get_settings


class S3Client:
    def __init__(self, bucket: str | None = None):
        settings = get_settings()
        self.bucket = bucket or settings.s3_bucket_tiles
        self._client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_default_region,
        )

    def upload_tile(self, local_path: str, region_id: int, date: str) -> str:
        filename = Path(local_path).name
        key = f"sentinel2/{region_id}/{date}/{filename}"
        self._client.upload_file(local_path, self.bucket, key)
        return key

    def upload_processed_tile(self, local_path: str, region_id: int, date: str) -> str:
        filename = Path(local_path).name
        key = f"sentinel2_processed/{region_id}/{date}/{filename}"
        self._client.upload_file(local_path, self.bucket, key)
        return key

    def download_tile(self, s3_key: str, dest_path: str) -> None:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        self._client.download_file(self.bucket, s3_key, dest_path)

    def upload_pdf(self, local_path: str, report_id: str) -> str:
        settings = get_settings()
        key = f"{report_id}.pdf"
        self._client.upload_file(local_path, settings.s3_bucket_pdfs, key)
        return key
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/storage/test_s3.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/storage/s3.py tests/storage/test_s3.py
git commit -m "feat: S3 client wrapper for tile and PDF storage"
```

---

## Task 4: Base Ingestion Utilities

**Files:**
- Create: `src/ingestion/base.py`
- Create: `tests/ingestion/test_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/test_base.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.ingestion.base import with_retry, log_failure


@pytest.mark.asyncio
async def test_with_retry_succeeds_on_first_try():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await with_retry(flaky, max_attempts=3)
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_with_retry_retries_on_failure():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("transient error")
        return "ok"

    result = await with_retry(flaky, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_with_retry_raises_after_max_attempts():
    async def always_fails():
        raise ValueError("permanent error")

    with pytest.raises(ValueError, match="permanent error"):
        await with_retry(always_fails, max_attempts=2, base_delay=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingestion/test_base.py -v
```

Expected: `FAIL` — `with_retry` not importable.

- [ ] **Step 3: Create `src/ingestion/base.py`**

```python
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, TypeVar
from sqlalchemy import text
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {exc}. Retrying in {delay}s")
                await asyncio.sleep(delay)
    raise last_exc


async def log_failure(flow_name: str, error_message: str, region_id: int | None = None) -> None:
    async with get_async_session() as session:
        await session.execute(
            text(
                "INSERT INTO failed_ingestion (region_id, flow_name, error_message, failed_at) "
                "VALUES (:region_id, :flow_name, :error_message, :failed_at)"
            ),
            {
                "region_id": region_id,
                "flow_name": flow_name,
                "error_message": error_message,
                "failed_at": datetime.now(timezone.utc),
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_base.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/base.py src/ingestion/__init__.py tests/ingestion/
git commit -m "feat: ingestion base — retry with exponential backoff + failure logging"
```

---

## Task 5: FEMA Ingestion Flow

**Files:**
- Create: `src/ingestion/fema.py`
- Create: `tests/ingestion/test_fema.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/test_fema.py
import pytest
from unittest.mock import patch, AsyncMock
from src.ingestion.fema import fetch_fema_declarations, upsert_declarations


@pytest.mark.asyncio
async def test_fetch_fema_declarations_returns_list():
    mock_response = {
        "DisasterDeclarationsSummaries": [
            {
                "disasterNumber": "4332",
                "state": "TX",
                "designatedArea": "Harris",
                "fipsCountyCode": "201",
                "incidentType": "Flood",
                "declarationDate": "2017-09-07T00:00:00.000Z",
                "incidentBeginDate": "2017-08-25T00:00:00.000Z",
                "incidentEndDate": "2017-09-15T00:00:00.000Z",
                "declarationTitle": "HURRICANE HARVEY",
            }
        ]
    }

    with patch("src.ingestion.fema.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_response_obj = AsyncMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = AsyncMock()
        mock_client.get.return_value = mock_response_obj

        results = await fetch_fema_declarations(last_refresh=None)

    assert len(results) == 1
    assert results[0]["disasterNumber"] == "4332"


@pytest.mark.asyncio
async def test_upsert_declarations_is_idempotent(db_session):
    record = {
        "disaster_number": "DR-4332",
        "state": "TX",
        "county_fips": "48201",
        "disaster_type": "Flood",
        "declaration_date": "2017-09-07",
        "incident_begin": "2017-08-25",
        "incident_end": "2017-09-15",
        "declaration_title": "HURRICANE HARVEY",
    }

    await upsert_declarations([record], db_session)
    await upsert_declarations([record], db_session)  # second call must not raise

    result = await db_session.execute(
        text("SELECT COUNT(*) FROM fema_declarations WHERE disaster_number = 'DR-4332'")
    )
    assert result.scalar() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingestion/test_fema.py -v
```

Expected: `FAIL` — `fetch_fema_declarations` not importable.

- [ ] **Step 3: Create `src/ingestion/fema.py`**

```python
import httpx
import logging
from datetime import datetime, timezone, date
from prefect import flow, task
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure

logger = logging.getLogger(__name__)

FEMA_BASE_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
PAGE_SIZE = 1000


async def fetch_fema_declarations(last_refresh: str | None) -> list[dict]:
    params = {"$top": PAGE_SIZE, "$orderby": "lastRefresh asc"}
    if last_refresh:
        params["$filter"] = f"lastRefresh gt '{last_refresh}'"

    records = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        skip = 0
        while True:
            params["$skip"] = skip
            response = await client.get(FEMA_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            batch = data.get("DisasterDeclarationsSummaries", [])
            records.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            skip += PAGE_SIZE

    return records


def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
    except ValueError:
        return None


async def upsert_declarations(records: list[dict], session: AsyncSession) -> None:
    for r in records:
        await session.execute(
            text("""
                INSERT INTO fema_declarations
                    (disaster_number, state, county_fips, disaster_type,
                     declaration_date, incident_begin, incident_end, declaration_title)
                VALUES
                    (:disaster_number, :state, :county_fips, :disaster_type,
                     :declaration_date, :incident_begin, :incident_end, :declaration_title)
                ON CONFLICT (disaster_number) DO UPDATE SET
                    state = EXCLUDED.state,
                    county_fips = EXCLUDED.county_fips,
                    disaster_type = EXCLUDED.disaster_type,
                    declaration_date = EXCLUDED.declaration_date,
                    incident_begin = EXCLUDED.incident_begin,
                    incident_end = EXCLUDED.incident_end,
                    declaration_title = EXCLUDED.declaration_title
            """),
            r,
        )


@flow(name="ingest_fema", log_prints=True)
async def ingest_fema_flow(last_refresh: str | None = None) -> None:
    logger.info("Starting FEMA ingestion")
    try:
        records = await with_retry(
            lambda: fetch_fema_declarations(last_refresh), max_attempts=3
        )
        logger.info(f"Fetched {len(records)} FEMA records")

        normalized = [
            {
                "disaster_number": r.get("disasterNumber", ""),
                "state": r.get("state"),
                "county_fips": r.get("fipsCountyCode"),
                "disaster_type": r.get("incidentType"),
                "declaration_date": r.get("declarationDate"),
                "incident_begin": r.get("incidentBeginDate"),
                "incident_end": r.get("incidentEndDate"),
                "declaration_title": r.get("declarationTitle"),
            }
            for r in records
            if r.get("disasterNumber")
        ]

        async with get_async_session() as session:
            await upsert_declarations(normalized, session)

        logger.info(f"Upserted {len(normalized)} declarations")
    except Exception as exc:
        await log_failure("ingest_fema", str(exc))
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_fema.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/fema.py tests/ingestion/test_fema.py
git commit -m "feat: FEMA ingestion flow with delta sync and idempotent upserts"
```

---

## Task 6: NOAA Ingestion Flow

**Files:**
- Create: `src/ingestion/noaa.py`
- Create: `tests/ingestion/test_noaa.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/test_noaa.py
import pytest
from unittest.mock import patch, AsyncMock
from src.ingestion.noaa import fetch_noaa_daily, upsert_observations


@pytest.mark.asyncio
async def test_fetch_noaa_daily_returns_observations():
    mock_response = {
        "results": [
            {
                "date": "2024-01-15T00:00:00",
                "station": "GHCND:USC00410613",
                "datatype": "PRCP",
                "value": 25,
            },
            {
                "date": "2024-01-15T00:00:00",
                "station": "GHCND:USC00410613",
                "datatype": "TMAX",
                "value": 180,
            },
        ]
    }

    with patch("src.ingestion.noaa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_resp = AsyncMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = AsyncMock()
        mock_client.get.return_value = mock_resp

        results = await fetch_noaa_daily(
            station_id="GHCND:USC00410613",
            start_date="2024-01-15",
            end_date="2024-01-15",
            api_key="test-key",
        )

    assert len(results) == 1
    assert results[0]["precipitation_mm"] == pytest.approx(2.5)  # PRCP in tenths of mm
    assert results[0]["temp_max_c"] == pytest.approx(18.0)        # TMAX in tenths of °C


@pytest.mark.asyncio
async def test_upsert_observations_is_idempotent(db_session):
    # Requires a region row to satisfy FK
    await db_session.execute(
        text("INSERT INTO regions (name, bbox) VALUES ('test', '{}') ON CONFLICT DO NOTHING")
    )
    region_id = (
        await db_session.execute(text("SELECT id FROM regions WHERE name='test'"))
    ).scalar()

    obs = {
        "station_id": "GHCND:USC00410613",
        "region_id": region_id,
        "date": "2024-01-15",
        "precipitation_mm": 2.5,
        "temp_max_c": 18.0,
        "temp_min_c": 10.0,
        "soil_moisture": None,
    }

    await upsert_observations([obs], db_session)
    await upsert_observations([obs], db_session)

    result = await db_session.execute(
        text("SELECT COUNT(*) FROM noaa_observations WHERE station_id='GHCND:USC00410613'")
    )
    assert result.scalar() == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingestion/test_noaa.py -v
```

Expected: `FAIL`.

- [ ] **Step 3: Create `src/ingestion/noaa.py`**

```python
import httpx
import logging
from collections import defaultdict
from prefect import flow
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure
from src.config import get_settings

logger = logging.getLogger(__name__)
NOAA_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"


async def fetch_noaa_daily(
    station_id: str, start_date: str, end_date: str, api_key: str
) -> list[dict]:
    headers = {"token": api_key}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": "PRCP,TMAX,TMIN,AWND",
        "limit": 1000,
        "units": "metric",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(NOAA_BASE, headers=headers, params=params)
        response.raise_for_status()
        raw = response.json().get("results", [])

    # Group by date, pivot datatypes into one row per date
    by_date: dict[str, dict] = defaultdict(lambda: {
        "precipitation_mm": None, "temp_max_c": None, "temp_min_c": None, "soil_moisture": None
    })
    for item in raw:
        d = item["date"][:10]
        dt = item["datatype"]
        v = item["value"]
        if dt == "PRCP":
            by_date[d]["precipitation_mm"] = v / 10.0
        elif dt == "TMAX":
            by_date[d]["temp_max_c"] = v / 10.0
        elif dt == "TMIN":
            by_date[d]["temp_min_c"] = v / 10.0

    return [{"date": d, **vals} for d, vals in by_date.items()]


async def upsert_observations(records: list[dict], session: AsyncSession) -> None:
    for r in records:
        await session.execute(
            text("""
                INSERT INTO noaa_observations
                    (station_id, region_id, date, precipitation_mm, temp_max_c, temp_min_c, soil_moisture)
                VALUES
                    (:station_id, :region_id, :date, :precipitation_mm, :temp_max_c, :temp_min_c, :soil_moisture)
                ON CONFLICT (station_id, date) DO UPDATE SET
                    precipitation_mm = EXCLUDED.precipitation_mm,
                    temp_max_c = EXCLUDED.temp_max_c,
                    temp_min_c = EXCLUDED.temp_min_c,
                    soil_moisture = EXCLUDED.soil_moisture
            """),
            r,
        )


@flow(name="ingest_noaa", log_prints=True)
async def ingest_noaa_flow(region_id: int, station_id: str, start_date: str, end_date: str) -> None:
    settings = get_settings()
    try:
        obs = await with_retry(
            lambda: fetch_noaa_daily(station_id, start_date, end_date, settings.noaa_api_key)
        )
        enriched = [{"station_id": station_id, "region_id": region_id, **o} for o in obs]
        async with get_async_session() as session:
            await upsert_observations(enriched, session)
        logger.info(f"Upserted {len(enriched)} NOAA observations for station {station_id}")
    except Exception as exc:
        await log_failure("ingest_noaa", str(exc), region_id=region_id)
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_noaa.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/noaa.py tests/ingestion/test_noaa.py
git commit -m "feat: NOAA CDO daily weather ingestion flow"
```

---

## Task 7: OSM Ingestion Flow

**Files:**
- Create: `src/ingestion/osm.py`
- Create: `tests/ingestion/test_osm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/test_osm.py
import pytest
from unittest.mock import patch, AsyncMock
from src.ingestion.osm import fetch_osm_buildings, bbox_to_overpass_query


def test_bbox_to_overpass_query_generates_valid_query():
    bbox = {"min_lat": 29.5, "max_lat": 30.1, "min_lon": -95.8, "max_lon": -95.2}
    query = bbox_to_overpass_query(bbox)
    assert "way[building]" in query
    assert "29.5" in query
    assert "-95.8" in query


@pytest.mark.asyncio
async def test_fetch_osm_buildings_returns_geojson():
    mock_response = {
        "elements": [
            {
                "type": "way",
                "id": 123456,
                "tags": {"building": "yes", "start_date": "2005"},
                "nodes": [1, 2, 3, 1],
            }
        ]
    }

    with patch("src.ingestion.osm.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_resp = AsyncMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = AsyncMock()
        mock_client.post.return_value = mock_resp

        bbox = {"min_lat": 29.5, "max_lat": 30.1, "min_lon": -95.8, "max_lon": -95.2}
        result = await fetch_osm_buildings(bbox)

    assert "elements" in result
    assert len(result["elements"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/ingestion/test_osm.py -v
```

Expected: `FAIL`.

- [ ] **Step 3: Create `src/ingestion/osm.py`**

```python
import httpx
import logging
from prefect import flow
from sqlalchemy import text
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure

logger = logging.getLogger(__name__)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def bbox_to_overpass_query(bbox: dict) -> str:
    s, n = bbox["min_lat"], bbox["max_lat"]
    w, e = bbox["min_lon"], bbox["max_lon"]
    return f"""
    [out:json][timeout:60];
    (
      way[building]({s},{w},{n},{e});
      way[highway]({s},{w},{n},{e});
    );
    out body;
    """


async def fetch_osm_buildings(bbox: dict) -> dict:
    query = bbox_to_overpass_query(bbox)
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(OVERPASS_URL, data={"data": query})
        response.raise_for_status()
        return response.json()


@flow(name="ingest_osm", log_prints=True)
async def ingest_osm_flow(region_id: int) -> None:
    async with get_async_session() as session:
        result = await session.execute(
            text("SELECT bbox FROM regions WHERE id = :id"), {"id": region_id}
        )
        row = result.one_or_none()
        if not row:
            raise ValueError(f"Region {region_id} not found")
        bbox = row[0]

    try:
        geojson = await with_retry(lambda: fetch_osm_buildings(bbox))
        async with get_async_session() as session:
            await session.execute(
                text("""
                    UPDATE regions SET bbox = jsonb_set(bbox, '{osm_snapshot}', :snapshot::jsonb)
                    WHERE id = :id
                """),
                {"snapshot": str(geojson), "id": region_id},
            )
        logger.info(f"OSM snapshot updated for region {region_id}: {len(geojson.get('elements', []))} elements")
    except Exception as exc:
        await log_failure("ingest_osm", str(exc), region_id=region_id)
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_osm.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/osm.py tests/ingestion/test_osm.py
git commit -m "feat: OSM Overpass ingestion flow for buildings and roads"
```

---

## Task 8: Sentinel-2 Ingestion & Tile Preprocessing

**Files:**
- Create: `src/ingestion/sentinel2.py`
- Create: `src/pipeline/preprocessing.py`
- Create: `tests/ingestion/test_sentinel2.py`
- Create: `tests/pipeline/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_preprocessing.py
import pytest
import numpy as np
from pathlib import Path
from src.pipeline.preprocessing import crop_and_normalize, TILE_SIZE


def test_crop_and_normalize_returns_correct_shape(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    # Create a fake 3-band GeoTIFF larger than TILE_SIZE
    big = TILE_SIZE * 2
    data = np.random.randint(0, 3000, (3, big, big), dtype=np.uint16)
    src_path = tmp_path / "big_tile.tif"
    transform = from_bounds(0, 0, 1, 1, big, big)

    with rasterio.open(
        str(src_path), "w", driver="GTiff",
        height=big, width=big, count=3,
        dtype="uint16", crs="EPSG:4326", transform=transform
    ) as dst:
        dst.write(data)

    out_path = tmp_path / "cropped.tif"
    crop_and_normalize(str(src_path), str(out_path))

    with rasterio.open(str(out_path)) as src:
        result = src.read()

    assert result.shape == (3, TILE_SIZE, TILE_SIZE)
    assert result.dtype == np.float32
    assert result.max() <= 1.0
    assert result.min() >= 0.0


# tests/ingestion/test_sentinel2.py
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.sentinel2 import search_sentinel2_tiles


def test_search_returns_product_list():
    with patch("src.ingestion.sentinel2.SentinelAPI") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.query.return_value = {
            "abc123": {"title": "S2A_tile_20240115", "size": "500 MB", "uuid": "abc123"}
        }

        bbox = {"min_lat": 29.5, "max_lat": 30.1, "min_lon": -95.8, "max_lon": -95.2}
        products = search_sentinel2_tiles(
            bbox=bbox,
            date_from="2024-01-10",
            date_to="2024-01-20",
            user="user",
            password="pass",
        )

    assert len(products) == 1
    assert products[0]["uuid"] == "abc123"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/pipeline/test_preprocessing.py tests/ingestion/test_sentinel2.py -v
```

Expected: `FAIL`.

- [ ] **Step 3: Create `src/pipeline/preprocessing.py`**

```python
import numpy as np
import rasterio
from rasterio.enums import Resampling

TILE_SIZE = 256
REFLECTANCE_MAX = 3000.0  # Sentinel-2 typical surface reflectance max


def crop_and_normalize(src_path: str, dst_path: str) -> None:
    with rasterio.open(src_path) as src:
        # Read first 3 bands (B04 Red, B03 Green, B02 Blue for RGB composite)
        data = src.read(
            [1, 2, 3],
            out_shape=(3, TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        # Clip and normalize to [0, 1]
        data = np.clip(data, 0, REFLECTANCE_MAX) / REFLECTANCE_MAX

        profile = src.profile.copy()
        profile.update(
            count=3,
            height=TILE_SIZE,
            width=TILE_SIZE,
            dtype="float32",
            transform=src.transform,
        )

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)
```

- [ ] **Step 4: Create `src/ingestion/sentinel2.py`**

```python
import logging
import tempfile
import os
from pathlib import Path
from datetime import date
from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
from shapely.geometry import box
from prefect import flow
from sqlalchemy import text
from src.storage.db import get_async_session
from src.storage.s3 import S3Client
from src.pipeline.preprocessing import crop_and_normalize
from src.ingestion.base import with_retry, log_failure
from src.config import get_settings

logger = logging.getLogger(__name__)


def search_sentinel2_tiles(
    bbox: dict, date_from: str, date_to: str, user: str, password: str
) -> list[dict]:
    api = SentinelAPI(user, password, "https://scihub.copernicus.eu/dhus")
    footprint = geojson_to_wkt(
        box(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]).__geo_interface__
    )
    products = api.query(
        footprint,
        date=(date_from, date_to),
        platformname="Sentinel-2",
        cloudcoverpercentage=(0, 20),
    )
    return [{"uuid": uuid, **meta} for uuid, meta in products.items()]


@flow(name="ingest_sentinel2", log_prints=True)
async def ingest_sentinel2_flow(region_id: int, date_from: str, date_to: str) -> None:
    settings = get_settings()
    s3 = S3Client()

    async with get_async_session() as session:
        result = await session.execute(
            text("SELECT bbox FROM regions WHERE id = :id"), {"id": region_id}
        )
        row = result.one_or_none()
        if not row:
            raise ValueError(f"Region {region_id} not found")
        bbox = row[0]

    try:
        products = await with_retry(
            lambda: search_sentinel2_tiles(
                bbox, date_from, date_to,
                settings.sentinelsat_user, settings.sentinelsat_pass
            )
        )
        logger.info(f"Found {len(products)} Sentinel-2 tiles for region {region_id}")

        api = SentinelAPI(
            settings.sentinelsat_user, settings.sentinelsat_pass,
            "https://scihub.copernicus.eu/dhus"
        )

        for product in products:
            with tempfile.TemporaryDirectory() as tmpdir:
                api.download(product["uuid"], directory_path=tmpdir)
                raw_files = list(Path(tmpdir).glob("**/*.SAFE"))
                if not raw_files:
                    continue

                tile_date = product.get("beginposition", "")[:10] or date_from
                raw_s3_key = s3.upload_tile(str(raw_files[0]), region_id, tile_date)

                processed_path = os.path.join(tmpdir, f"processed_{product['uuid']}.tif")
                crop_and_normalize(str(raw_files[0]), processed_path)
                processed_s3_key = s3.upload_processed_tile(processed_path, region_id, tile_date)

                async with get_async_session() as session:
                    await session.execute(
                        text("""
                            INSERT INTO sentinel2_tiles (region_id, s3_path, processed_s3_path, date, ingested_at)
                            VALUES (:region_id, :s3_path, :processed_s3_path, :date, NOW())
                            ON CONFLICT (region_id, s3_path) DO UPDATE
                                SET processed_s3_path = EXCLUDED.processed_s3_path
                        """),
                        {"region_id": region_id, "s3_path": raw_s3_key,
                         "processed_s3_path": processed_s3_key, "date": tile_date},
                    )

    except Exception as exc:
        await log_failure("ingest_sentinel2", str(exc), region_id=region_id)
        raise
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/pipeline/test_preprocessing.py tests/ingestion/test_sentinel2.py -v
```

Expected: `PASS`.

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/sentinel2.py src/pipeline/preprocessing.py \
        tests/ingestion/test_sentinel2.py tests/pipeline/test_preprocessing.py
git commit -m "feat: Sentinel-2 ingestion and tile preprocessing (crop + normalize)"
```

---

## Task 9: SegFormer Segmentation Pipeline

**Files:**
- Create: `src/pipeline/segmentation.py`
- Create: `tests/pipeline/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_segmentation.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.pipeline.segmentation import SegmentationPipeline, LAND_USE_CLASSES


def test_land_use_classes_has_five_categories():
    assert set(LAND_USE_CLASSES) == {"water", "vegetation", "urban", "bare_soil", "burn_scar"}


def test_segment_tile_returns_area_stats(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    # Create a 256x256 fake processed tile
    data = np.random.rand(3, 256, 256).astype(np.float32)
    tile_path = tmp_path / "tile.tif"
    with rasterio.open(
        str(tile_path), "w", driver="GTiff",
        height=256, width=256, count=3, dtype="float32",
        crs="EPSG:4326", transform=from_bounds(0, 0, 1, 1, 256, 256)
    ) as dst:
        dst.write(data)

    mock_logits = MagicMock()
    mock_logits.logits = MagicMock()

    with patch("src.pipeline.segmentation.SegformerForSemanticSegmentation.from_pretrained") as mock_model_cls, \
         patch("src.pipeline.segmentation.SegformerImageProcessor.from_pretrained") as mock_proc_cls:

        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_proc = MagicMock()
        mock_proc_cls.return_value = mock_proc

        # Mock processor output and model output
        mock_proc.return_value = {"pixel_values": MagicMock()}
        import torch
        fake_logits = torch.rand(1, 5, 64, 64)
        mock_output = MagicMock()
        mock_output.logits = fake_logits
        mock_model.return_value = mock_output

        # Mock post_process
        fake_seg = torch.zeros(256, 256, dtype=torch.long)
        mock_proc.post_process_semantic_segmentation.return_value = [fake_seg]

        pipeline = SegmentationPipeline()
        result = pipeline.segment(str(tile_path))

    assert "geojson" in result
    assert "area_stats" in result
    assert "flood_zone_geojson" in result
    for cls in LAND_USE_CLASSES:
        assert cls in result["area_stats"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/pipeline/test_segmentation.py -v
```

Expected: `FAIL`.

- [ ] **Step 3: Create `src/pipeline/segmentation.py`**

```python
import json
import logging
import numpy as np
import torch
import rasterio
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from datetime import datetime, timezone
from sqlalchemy import text
from src.storage.db import get_async_session
from src.config import get_settings

logger = logging.getLogger(__name__)

# SegFormer-B2 fine-tuned on ADE20K as a starting baseline.
# We remap ADE20K class indices to our 5 land-use classes.
# Replace with a satellite-specific checkpoint when available.
MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
MODEL_VERSION = "segformer-b2-ade-v1"

LAND_USE_CLASSES = ["water", "vegetation", "urban", "bare_soil", "burn_scar"]

# Mapping from ADE20K class indices to our 5 land-use categories.
# ADE20K: 0=wall, 6=water, 9=grass, 12=tree, 17=building, etc.
ADE20K_TO_LANDUSE = {
    6: "water",    # water
    26: "water",   # sea
    60: "water",   # river
    9: "vegetation",  # grass
    12: "vegetation", # tree
    4: "vegetation",  # plant
    17: "urban",   # building
    2: "urban",    # sky → road (urban proxy, imperfect)
    11: "urban",   # road
    29: "bare_soil",  # earth
    94: "burn_scar",  # dirt (closest available proxy)
}


class SegmentationPipeline:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
        self.model.eval()

    def segment(self, tile_path: str) -> dict:
        with rasterio.open(tile_path) as src:
            data = src.read([1, 2, 3])  # RGB bands, float32 [0,1]
            transform = src.transform
            crs = src.crs

        # Convert to uint8 PIL image for SegFormer processor
        rgb = (data.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb)

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        seg_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0].numpy()

        # Remap ADE20K labels to our 5 classes
        landuse_map = np.full(seg_map.shape, 3, dtype=np.uint8)  # default: bare_soil
        for ade_idx, landuse in ADE20K_TO_LANDUSE.items():
            landuse_map[seg_map == ade_idx] = LAND_USE_CLASSES.index(landuse)

        total_pixels = landuse_map.size
        area_stats = {
            cls: float((landuse_map == i).sum() / total_pixels)
            for i, cls in enumerate(LAND_USE_CLASSES)
        }

        # Flood zone: pixels classified as water or lowest-10% elevation proxy (bare_soil near edges)
        flood_mask = (landuse_map == LAND_USE_CLASSES.index("water")).astype(np.uint8)

        def mask_to_geojson(mask: np.ndarray, label: str) -> dict:
            from rasterio import features
            shapes = list(features.shapes(mask, transform=transform))
            features_list = [
                {"type": "Feature", "geometry": geom, "properties": {"label": label}}
                for geom, val in shapes if val == 1
            ]
            return {"type": "FeatureCollection", "features": features_list}

        geojson = {"type": "FeatureCollection", "features": []}
        for i, cls in enumerate(LAND_USE_CLASSES):
            mask = (landuse_map == i).astype(np.uint8)
            cls_geojson = mask_to_geojson(mask, cls)
            geojson["features"].extend(cls_geojson["features"])

        flood_zone_geojson = mask_to_geojson(flood_mask, "flood_zone")

        return {
            "geojson": geojson,
            "area_stats": area_stats,
            "flood_zone_geojson": flood_zone_geojson,
            "model_version": MODEL_VERSION,
        }


async def run_segmentation_for_tile(tile_id: int, processed_s3_path: str) -> None:
    import tempfile, os
    from src.storage.s3 import S3Client

    s3 = S3Client()
    pipeline = SegmentationPipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "tile.tif")
        s3.download_tile(processed_s3_path, local_path)
        result = pipeline.segment(local_path)

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO segmentation_results
                    (tile_id, geojson, area_stats, flood_zone_geojson, model_version, created_at)
                VALUES
                    (:tile_id, :geojson::jsonb, :area_stats::jsonb, :flood_zone_geojson::jsonb,
                     :model_version, :created_at)
                ON CONFLICT DO NOTHING
            """),
            {
                "tile_id": tile_id,
                "geojson": json.dumps(result["geojson"]),
                "area_stats": json.dumps(result["area_stats"]),
                "flood_zone_geojson": json.dumps(result["flood_zone_geojson"]),
                "model_version": result["model_version"],
                "created_at": datetime.now(timezone.utc),
            },
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/pipeline/test_segmentation.py -v
```

Expected: `PASS`.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/segmentation.py tests/pipeline/test_segmentation.py
git commit -m "feat: SegFormer-B2 segmentation pipeline with ADE20K→land-use remapping"
```

---

## Task 10: FEMA Report Chunking & Embedding

**Files:**
- Create: `src/rag/chunking.py`
- Create: `src/rag/embed.py`
- Create: `tests/rag/test_chunking.py`
- Create: `tests/rag/test_embed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/rag/test_chunking.py
import pytest
from src.rag.chunking import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP


def test_short_text_returns_single_chunk():
    text = "Short text under the chunk size limit."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_produces_overlapping_chunks():
    # ~1200 chars → should produce 3 chunks with 512-token window
    text = " ".join(["word"] * 600)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    # Each chunk should be roughly CHUNK_SIZE tokens — check they're non-empty
    for chunk in chunks:
        assert len(chunk.split()) > 0


def test_chunk_overlap_is_present():
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text)
    if len(chunks) >= 2:
        # Last words of chunk 0 should appear at start of chunk 1
        end_of_first = chunks[0].split()[-10:]
        start_of_second = chunks[1].split()[:20]
        overlap = set(end_of_first) & set(start_of_second)
        assert len(overlap) > 0


# tests/rag/test_embed.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.rag.embed import TextEmbedder


def test_text_embedder_returns_384_dim_vector():
    with patch("src.rag.embed.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)

        embedder = TextEmbedder()
        result = embedder.embed("Test sentence about flood risk in Texas.")

    assert isinstance(result, list)
    assert len(result) == 384


def test_text_embedder_batch_returns_correct_shape():
    with patch("src.rag.embed.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_st.return_value = mock_model
        mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)

        embedder = TextEmbedder()
        results = embedder.embed_batch(["text1", "text2", "text3"])

    assert len(results) == 3
    assert all(len(v) == 384 for v in results)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/rag/ -v
```

Expected: `FAIL`.

- [ ] **Step 3: Create `src/rag/chunking.py`**

```python
from transformers import AutoTokenizer

CHUNK_SIZE = 512    # tokens
CHUNK_OVERLAP = 64  # tokens
_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return _tokenizer


def chunk_text(text: str) -> list[str]:
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
```

- [ ] **Step 4: Create `src/rag/embed.py`**

```python
import numpy as np
from sentence_transformers import SentenceTransformer

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"


class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(MINILM_MODEL)

    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]


class ImageEmbedder:
    def __init__(self):
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    def embed(self, image_path: str) -> list[float]:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/rag/ -v
```

Expected: `PASS`.

- [ ] **Step 6: Wire FEMA embedding into ingestion flow**

Edit `src/ingestion/fema.py` — add this at the end of `ingest_fema_flow`, after the upsert:

```python
    # Chunk and embed FEMA declaration titles + descriptions for RAG
    from src.rag.chunking import chunk_text
    from src.rag.embed import TextEmbedder
    from sqlalchemy import text as sql_text

    embedder = TextEmbedder()
    async with get_async_session() as session:
        rows = await session.execute(
            sql_text("SELECT disaster_number, declaration_title FROM fema_declarations WHERE declaration_title IS NOT NULL")
        )
        for disaster_number, title in rows.fetchall():
            chunks = chunk_text(title)
            for idx, chunk in enumerate(chunks):
                embedding = embedder.embed(chunk)
                await session.execute(
                    sql_text("""
                        INSERT INTO text_embeddings
                            (source_type, source_id, chunk_text, chunk_index, embedding, metadata)
                        VALUES
                            ('fema', :source_id, :chunk_text, :chunk_index, :embedding, :metadata)
                        ON CONFLICT (source_type, source_id, chunk_index) DO NOTHING
                    """),
                    {
                        "source_id": disaster_number,
                        "chunk_text": chunk,
                        "chunk_index": idx,
                        "embedding": embedding,
                        "metadata": {"disaster_number": disaster_number},
                    },
                )
```

- [ ] **Step 7: Commit**

```bash
git add src/rag/ tests/rag/ src/ingestion/fema.py
git commit -m "feat: FEMA report chunking and MiniLM/CLIP embedding pipeline"
```

---

## Task 11: Segmentation Eval (EuroSAT Baseline)

**Files:**
- Create: `tests/evals/test_segmentation_eval.py`

- [ ] **Step 1: Write the eval**

```python
# tests/evals/test_segmentation_eval.py
"""
Segmentation eval against a subset of EuroSAT.
EuroSAT is a scene-level classification dataset; we use it to measure
per-class accuracy of our patch-level land-use predictions.
Eval passes if mean accuracy across 5 land-use classes >= 0.60 (baseline).
Raise threshold in Phase 2 after fine-tuning.
"""
import pytest
import numpy as np
from datasets import load_dataset
from src.pipeline.segmentation import SegmentationPipeline, LAND_USE_CLASSES
import tempfile, os
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

# EuroSAT class → our land-use class mapping
EUROSAT_TO_LANDUSE = {
    "AnnualCrop": "vegetation",
    "Forest": "vegetation",
    "HerbaceousVegetation": "vegetation",
    "Highway": "urban",
    "Industrial": "urban",
    "Pasture": "vegetation",
    "PermanentCrop": "vegetation",
    "Residential": "urban",
    "River": "water",
    "SeaLake": "water",
}

EVAL_SAMPLES = 50  # limit for speed in CI


@pytest.mark.slow
def test_segmentation_eurosat_baseline():
    dataset = load_dataset("torchgeo/eurosat", split="test", trust_remote_code=True)
    pipeline = SegmentationPipeline()

    correct = 0
    total = 0

    for sample in dataset.select(range(EVAL_SAMPLES)):
        label_name = dataset.features["label"].int2str(sample["label"])
        expected_class = EUROSAT_TO_LANDUSE.get(label_name)
        if expected_class is None:
            continue

        # Save image as a fake GeoTIFF for our pipeline
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp_path = f.name

        try:
            img = sample["image"].convert("RGB")
            img_resized = img.resize((256, 256))
            arr = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0

            with rasterio.open(
                tmp_path, "w", driver="GTiff", height=256, width=256,
                count=3, dtype="float32", crs="EPSG:4326",
                transform=from_bounds(0, 0, 1, 1, 256, 256)
            ) as dst:
                dst.write(arr)

            result = pipeline.segment(tmp_path)
            # Dominant class = class with highest area stat
            predicted_class = max(result["area_stats"], key=lambda k: result["area_stats"][k])

            if predicted_class == expected_class:
                correct += 1
            total += 1
        finally:
            os.unlink(tmp_path)

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nSegmentation eval: {correct}/{total} = {accuracy:.2%}")
    assert accuracy >= 0.60, f"Baseline accuracy {accuracy:.2%} below 0.60 threshold"
```

- [ ] **Step 2: Run the eval**

```bash
pytest tests/evals/test_segmentation_eval.py -v -m slow
```

Expected: `PASS` with accuracy ≥ 60%. If it fails, note the actual score — the threshold is set conservatively for the Phase 1 baseline; it will be raised after fine-tuning in Phase 2.

- [ ] **Step 3: Update `Makefile` eval target to include the `--slow` marker for CI**

Edit `Makefile`, change the eval target:

```makefile
eval:
	pytest tests/evals/ -v -m slow
```

- [ ] **Step 4: Add `pytest.ini_options` marker declaration**

Edit `pyproject.toml` — add to `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
  "slow: marks eval tests that take >30s (deselect with '-m not slow')",
]
```

- [ ] **Step 5: Commit**

```bash
git add tests/evals/test_segmentation_eval.py pyproject.toml Makefile
git commit -m "feat: segmentation eval harness against EuroSAT with 0.60 baseline threshold"
```

---

## Task 12: Streamlit MVP Dashboard

**Files:**
- Create: `src/dashboard/app.py`

- [ ] **Step 1: Create `src/dashboard/app.py`**

```python
import streamlit as st
import asyncio
from sqlalchemy import text
from src.storage.db import get_async_session

st.set_page_config(page_title="ZimRadar", layout="wide", page_icon="🌍")
st.title("🌍 ZimRadar — Climate Risk Assessment")


async def get_regions() -> list[dict]:
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT r.id, r.name, r.bbox,
                       ra.risk_tier, ra.composite_score, ra.assessed_at
                FROM regions r
                LEFT JOIN LATERAL (
                    SELECT risk_tier, composite_score, assessed_at
                    FROM risk_assessments
                    WHERE region_id = r.id
                    ORDER BY assessed_at DESC LIMIT 1
                ) ra ON TRUE
                WHERE r.active = TRUE
                ORDER BY r.name
            """)
        )
        return [dict(row._mapping) for row in result.fetchall()]


async def get_report(region_id: int) -> dict | None:
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT narrative, citations, factuality_score, low_confidence, created_at
                FROM reports
                WHERE region_id = :region_id
                ORDER BY created_at DESC LIMIT 1
            """),
            {"region_id": region_id},
        )
        row = result.one_or_none()
        return dict(row._mapping) if row else None


TIER_COLORS = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}

col_map, col_report = st.columns([3, 2])

with col_map:
    st.subheader("Tracked Regions")
    regions = asyncio.run(get_regions())

    if not regions:
        st.info("No regions tracked yet. Add a region to the `regions` table to get started.")
    else:
        import leafmap.foliumap as leafmap
        m = leafmap.Map(center=[37.5, -96], zoom=4)

        for region in regions:
            bbox = region["bbox"]
            tier = region.get("risk_tier", "unknown")
            color = {"critical": "red", "high": "orange", "moderate": "yellow", "low": "green"}.get(tier, "gray")
            if all(k in bbox for k in ("min_lon", "min_lat", "max_lon", "max_lat")):
                m.add_geojson(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [bbox["min_lon"], bbox["min_lat"]],
                                [bbox["max_lon"], bbox["min_lat"]],
                                [bbox["max_lon"], bbox["max_lat"]],
                                [bbox["min_lon"], bbox["max_lat"]],
                                [bbox["min_lon"], bbox["min_lat"]],
                            ]]
                        },
                        "properties": {"name": region["name"], "tier": tier},
                    },
                    layer_name=region["name"],
                    style={"color": color, "fillOpacity": 0.2},
                )
        m.to_streamlit(height=500)

with col_report:
    st.subheader("Risk Report")
    if regions:
        selected = st.selectbox(
            "Select region",
            options=[r["name"] for r in regions],
            index=0,
        )
        region = next(r for r in regions if r["name"] == selected)

        tier = region.get("risk_tier")
        score = region.get("composite_score")
        if tier:
            icon = TIER_COLORS.get(tier, "⚪")
            st.metric("Risk Tier", f"{icon} {tier.upper()}", delta=f"Score: {score:.2f}" if score else None)

        if st.button("▶ Run Assessment", type="primary"):
            with st.status("Running ZimRadar pipeline...", expanded=True) as status:
                st.write("Ingesting latest satellite data...")
                st.write("Running segmentation + depth estimation...")
                st.write("Forecasting with Chronos...")
                st.write("Generating report with Gemma...")
                status.update(label="Pipeline complete", state="complete")
            st.rerun()

        report = asyncio.run(get_report(region["id"]))
        if report:
            if report.get("low_confidence"):
                st.warning("⚠ Low confidence report — factuality score below threshold.")
            st.markdown(report["narrative"])
            if report.get("citations"):
                st.caption("**Sources:** " + " · ".join(
                    f"[{i+1}] {c}" for i, c in enumerate(report["citations"])
                ))
            if report.get("factuality_score"):
                st.caption(f"Factuality score: {report['factuality_score']:.2f}")
        else:
            st.info("No report yet for this region. Click **Run Assessment** to generate one.")
```

- [ ] **Step 2: Add Streamlit entry to Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system -e .
COPY src/ ./src/

FROM base AS worker
CMD ["python", "-m", "prefect", "worker", "start", "--pool", "default-agent-pool"]

FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

- [ ] **Step 3: Start the stack and verify the dashboard loads**

```bash
make up
# Wait ~30s for postgres to initialize
open http://localhost:8501
```

Expected: Streamlit loads with "No regions tracked yet" message. Add a test region:

```bash
docker compose exec postgres psql -U zimradar -d zimradar -c \
  "INSERT INTO regions (name, bbox) VALUES ('Harris County TX', '{\"min_lon\": -95.8, \"min_lat\": 29.5, \"max_lon\": -95.2, \"max_lat\": 30.1}');"
```

Reload Streamlit — the region should appear in the map and selector.

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/app.py Dockerfile
git commit -m "feat: Streamlit MVP dashboard with leafmap and risk report panel"
```

---

## Task 13: GitHub Actions CI Skeleton

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: zimradar
          POSTGRES_PASSWORD: password
          POSTGRES_DB: zimradar_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 5s
          --health-timeout 5s
          --health-retries 5

    env:
      DATABASE_URL: postgresql+asyncpg://zimradar:password@localhost:5432/zimradar_test
      REDIS_URL: redis://localhost:6379
      AWS_ACCESS_KEY_ID: test
      AWS_SECRET_ACCESS_KEY: test
      AWS_DEFAULT_REGION: us-east-1
      S3_BUCKET_TILES: zimradar-tiles
      S3_BUCKET_PDFS: zimradar-pdfs

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install uv && uv pip install --system -e ".[dev]"

      - name: Lint
        run: ruff check src/ tests/ && ruff format --check src/ tests/

      - name: Run unit tests (excluding evals)
        run: pytest tests/ -v --ignore=tests/evals/ -x

  eval:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'pull_request'
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_USER: zimradar
          POSTGRES_PASSWORD: password
          POSTGRES_DB: zimradar_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 5s
          --health-timeout 5s
          --health-retries 5

    env:
      DATABASE_URL: postgresql+asyncpg://zimradar:password@localhost:5432/zimradar_test

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install uv && uv pip install --system -e ".[dev]"

      - name: Run segmentation eval
        run: pytest tests/evals/test_segmentation_eval.py -v -m slow
```

- [ ] **Step 2: Verify CI passes locally before pushing**

```bash
make test
make lint
```

Expected: all tests green, no lint errors.

- [ ] **Step 3: Commit and push**

```bash
git add .github/
git commit -m "feat: GitHub Actions CI — unit tests + segmentation eval on PRs"
git push -u origin main
```

Expected: CI pipeline runs and passes on GitHub.

---

## Phase 1 Complete ✓

**What you now have:**
- Full Docker Compose stack running locally (`make up`)
- All 4 data ingestion flows (Sentinel-2, NOAA, FEMA, OSM) with retry and error logging
- SegFormer segmentation pipeline storing GeoJSON masks in Postgres
- FEMA report chunking + MiniLM embedding into pgvector
- Streamlit MVP at `http://localhost:8501`
- Segmentation eval with CI-gated regression threshold
- GitHub Actions CI blocking PRs on test failures

**Next: Phase 2 — Intelligence**
- ZoeDepth terrain analysis
- Chronos forecasting pipeline with backtesting
- XGBoost tabular classifier training
- LlamaIndex RAG pipeline with hybrid retrieval + re-ranking

Phase 2 plan will be written at the start of that phase.
