-- Fix FEMA schema: disaster_number is NOT globally unique —
-- each disaster can span many counties. Change to composite key
-- (disaster_number, county_fips) and re-ingest from scratch.

-- 1. Clear stale FEMA data (embeddings + declarations)
DELETE FROM text_embeddings WHERE source_type = 'fema';
TRUNCATE TABLE fema_declarations RESTART IDENTITY CASCADE;

-- 2. Drop the old single-column unique constraint
ALTER TABLE fema_declarations
    DROP CONSTRAINT IF EXISTS fema_declarations_disaster_number_key;

-- 3. Add composite unique key
ALTER TABLE fema_declarations
    ADD CONSTRAINT fema_declarations_number_fips_uidx
    UNIQUE (disaster_number, county_fips);

-- 4. Remove NOT NULL from disaster_number (county_fips can be NULL for state-level decl.)
ALTER TABLE fema_declarations
    ALTER COLUMN disaster_number DROP NOT NULL;
