-- Migration 008: county-level elevation summary (SRTM via USGS EPQS)
CREATE TABLE IF NOT EXISTS county_elevation_summary (
    county_fips       TEXT PRIMARY KEY,
    elevation_mean_m  FLOAT,
    elevation_std_m   FLOAT,
    updated_at        TIMESTAMPTZ DEFAULT now()
);
