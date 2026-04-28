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
