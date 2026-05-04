CREATE TABLE IF NOT EXISTS county_reports (
    id              SERIAL PRIMARY KEY,
    region_id       INTEGER REFERENCES regions(id),
    county_fips     VARCHAR(5),
    risk_tier       VARCHAR(20),
    confidence      FLOAT,
    top_drivers     JSONB,
    evidence        JSONB,
    briefing_md     TEXT,
    citations       JSONB,
    validation_pass BOOLEAN DEFAULT false,
    flagged         BOOLEAN DEFAULT false,
    model_version   VARCHAR(50),
    prompt_version  VARCHAR(10) DEFAULT 'v1',
    generated_at    TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_county_reports_region ON county_reports(region_id);
CREATE INDEX IF NOT EXISTS idx_county_reports_fips ON county_reports(county_fips);
