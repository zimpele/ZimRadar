-- Migration 009: county-level infrastructure age summary (OSM building start_date)
CREATE TABLE IF NOT EXISTS county_infrastructure_summary (
    county_fips           TEXT PRIMARY KEY,
    median_building_age_yr FLOAT,
    building_count        INT,
    updated_at            TIMESTAMPTZ DEFAULT now()
);
