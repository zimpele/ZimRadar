CREATE TABLE IF NOT EXISTS county_storm_summary (
    county_fips     VARCHAR(5) PRIMARY KEY,
    storm_events_5yr        INTEGER   DEFAULT 0,
    flood_events_5yr_noaa   INTEGER   DEFAULT 0,
    storm_damage_usd        FLOAT     DEFAULT 0.0,
    population              INTEGER   DEFAULT 0,
    storm_damage_per_capita FLOAT     DEFAULT 0.0,
    updated_at              TIMESTAMP DEFAULT now()
);
