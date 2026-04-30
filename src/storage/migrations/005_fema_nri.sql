-- Migration 005: FEMA National Risk Index county table
CREATE TABLE IF NOT EXISTS fema_nri_county (
    county_fips       TEXT PRIMARY KEY,
    state_abbr        TEXT,
    county_name       TEXT,
    risk_score        FLOAT,
    risk_rating       TEXT,
    eal_score         FLOAT,
    sovi_score        FLOAT,
    resl_score        FLOAT,
    -- hazard-specific risk scores
    cfld_risks        FLOAT,   -- coastal flood
    rfld_risks        FLOAT,   -- riverine flood
    hwav_risks        FLOAT,   -- heat wave
    drgt_risks        FLOAT,   -- drought
    wfir_risks        FLOAT,   -- wildfire
    swnd_risks        FLOAT,   -- strong wind
    trnd_risks        FLOAT,   -- tornado
    updated_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fema_nri_state ON fema_nri_county (state_abbr);
