ALTER TABLE regions
    ADD COLUMN IF NOT EXISTS state_code  TEXT,
    ADD COLUMN IF NOT EXISTS county_fips TEXT,
    ADD COLUMN IF NOT EXISTS geometry    JSONB;

-- Prevent duplicate county entries; NULL FIPS (e.g. BW region) are excluded.
CREATE UNIQUE INDEX IF NOT EXISTS regions_county_fips_uidx
    ON regions (county_fips)
    WHERE county_fips IS NOT NULL;
