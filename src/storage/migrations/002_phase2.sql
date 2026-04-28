CREATE TABLE IF NOT EXISTS depth_results (
    id SERIAL PRIMARY KEY,
    tile_id INTEGER NOT NULL REFERENCES sentinel2_tiles(id) ON DELETE CASCADE,
    flood_zone_geojson JSONB,
    depth_map_s3_path TEXT,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tile_id, model_version)
);
