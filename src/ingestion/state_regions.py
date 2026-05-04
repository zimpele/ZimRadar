"""Prefect flows: seed county regions for US states and bulk-ingest Sentinel-2."""

import asyncio
import logging
from datetime import date, timedelta

from prefect import flow, task, get_run_logger
from sqlalchemy import text

from src.ingestion.base import log_failure
from src.ingestion.geo_admin import add_county_region, list_counties_for_state
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

SENTINEL2_CONCURRENCY = 2  # conservative — Copernicus throttles heavily


@task(name="seed-counties-for-states", log_prints=True)
async def seed_counties_for_states(state_codes: list[str]) -> list[int]:
    log = get_run_logger()
    region_ids: list[int] = []
    for state_code in state_codes:
        counties = await list_counties_for_state(state_code)
        log.info("State %s: %d counties", state_code, len(counties))
        for county_name, fips in counties:
            try:
                rid = await add_county_region(fips)
                region_ids.append(rid)
            except Exception as exc:
                log.warning("FIPS %s (%s) skipped: %s", fips, county_name, exc)
    log.info("Seeded %d regions for states: %s", len(region_ids), state_codes)
    return region_ids


@flow(name="seed_state_regions", log_prints=True)
async def seed_state_regions_flow(state_codes: list[str] = ["CA", "FL"]) -> int:
    """Create a region entry for every county in the given US states.

    Downloads Census county GeoJSON once, then bulk-inserts into the regions
    table.  Safe to re-run — existing counties are skipped.
    """
    logger.info("Seeding county regions for states: %s", state_codes)
    try:
        region_ids = await seed_counties_for_states(state_codes)
        logger.info("Done — %d regions created/verified", len(region_ids))
        return len(region_ids)
    except Exception as exc:
        await log_failure("seed_state_regions", str(exc))
        raise


@flow(name="bulk_ingest_sentinel2_states", log_prints=True)
async def bulk_ingest_sentinel2_flow(
    state_codes: list[str] = ["CA", "FL"],
    date_from: str | None = None,
    date_to: str | None = None,
) -> int:
    """Ingest Sentinel-2 tiles for all county regions in the given US states.

    Run seed_state_regions first to populate the regions table.
    Defaults to the last 90 days of imagery.
    """
    from src.ingestion.sentinel2 import ingest_sentinel2_flow

    if not date_to:
        date_to = date.today().isoformat()
    if not date_from:
        date_from = (date.today() - timedelta(days=90)).isoformat()

    logger.info("Bulk Sentinel-2 for %s: %s → %s", state_codes, date_from, date_to)

    async with get_async_session() as session:
        rows = await session.execute(
            text("""
                SELECT id, name FROM regions
                WHERE state_code = ANY(:states) AND county_fips IS NOT NULL
                ORDER BY id
            """),
            {"states": state_codes},
        )
        regions = rows.fetchall()

    if not regions:
        logger.warning("No regions found for %s — run seed_state_regions first.", state_codes)
        return 0

    logger.info(
        "Ingesting Sentinel-2 for %d regions (concurrency=%d)", len(regions), SENTINEL2_CONCURRENCY
    )
    sem = asyncio.Semaphore(SENTINEL2_CONCURRENCY)
    ok = 0
    errors = 0

    async def ingest_one(region_id: int, region_name: str) -> None:
        nonlocal ok, errors
        async with sem:
            try:
                await ingest_sentinel2_flow(region_id, date_from, date_to)
                ok += 1
            except Exception as exc:
                logger.warning("Region %d (%s) failed: %s", region_id, region_name, exc)
                errors += 1

    try:
        await asyncio.gather(*[ingest_one(rid, name) for rid, name in regions])
        logger.info(
            "Sentinel-2 bulk ingest complete — %d ok, %d errors out of %d",
            ok,
            errors,
            len(regions),
        )
        return ok
    except Exception as exc:
        await log_failure("bulk_ingest_sentinel2_states", str(exc))
        raise
