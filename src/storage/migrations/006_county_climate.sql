-- Migration 006: county-level climate summary for all FEMA counties
CREATE TABLE IF NOT EXISTS county_climate_summary (
    county_fips   TEXT PRIMARY KEY,
    station_id    TEXT,
    avg_precip_mm FLOAT,
    precip_trend  FLOAT,   -- mm/day linear slope over fetch window
    obs_days      INT,
    updated_at    TIMESTAMPTZ DEFAULT now()
);
