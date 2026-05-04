"""Prefect flow: OSM buildings → infrastructure age / development score per county."""

import asyncio
import logging
import math
import re
from datetime import date
from statistics import median

import httpx
from prefect import flow, task, get_run_logger
from sqlalchemy import text

from src.ingestion.base import log_failure
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CONCURRENCY = 2
CURRENT_YEAR = date.today().year
MIN_YEAR = 1800
# OSM tags that may hold a construction year, tried in order
AGE_TAGS = ("start_date", "year_of_construction", "construction_date", "building:year", "built")


def _parse_year(s: str) -> int | None:
    """Extract a four-digit construction year from an OSM tag value.

    Handles: "1920", "1920-01-01", "ca. 1920", "~1950", "circa 1900".
    Returns None for unparseable or out-of-range values.
    """
    if not s:
        return None
    match = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", s)
    if not match:
        return None
    year = int(match.group(1))
    if year < MIN_YEAR or year > CURRENT_YEAR:
        return None
    return year


def _development_score(building_count: int) -> float:
    """Map building count to a pseudo-age proxy in the 0-50 range.

    Used when no explicit construction year tags are present (the common case
    for US OSM data).  More buildings → more developed area → higher score.
    log1p keeps small counts meaningful without letting dense urban areas
    dominate.  Calibrated so ~1000 buildings ≈ 50 (saturates at city scale).
    """
    return min(50.0, math.log1p(building_count) * 7.2)


async def _fetch_buildings(
    bbox: dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> list[dict]:
    """Fetch all building elements from Overpass for a county bbox."""
    min_lat = float(bbox["min_lat"])
    max_lat = float(bbox["max_lat"])
    min_lon = float(bbox["min_lon"])
    max_lon = float(bbox["max_lon"])
    overpass_bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query = (
        f"[out:json][timeout:90];"
        f'(way["building"]({overpass_bbox});'
        f'relation["building"]({overpass_bbox}););'
        f"out tags;"
    )

    async with sem:
        try:
            resp = await client.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception as exc:
            logger.debug("Overpass fetch failed: %s", exc)
            return []


async def _process_county(
    county_fips: str,
    bbox: dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Return infrastructure summary dict, or None on total fetch failure."""
    elements = await _fetch_buildings(bbox, client, sem)
    if not elements:
        return None

    years = []
    for el in elements:
        tags = el.get("tags", {})
        for tag in AGE_TAGS:
            year = _parse_year(tags.get(tag, ""))
            if year is not None:
                years.append(year)
                break  # first matching tag wins per element

    if years:
        age = float(CURRENT_YEAR - int(median(years)))
    else:
        # No explicit age data — use building count as development proxy
        age = _development_score(len(elements))

    return {
        "county_fips": county_fips,
        "median_building_age_yr": age,
        "building_count": len(elements),
    }


@task(name="build-county-infrastructure", log_prints=True)
async def build_county_infrastructure(skip_existing: bool = True) -> int:
    log = get_run_logger()

    async with get_async_session() as session:
        if skip_existing:
            sql = """
                SELECT r.county_fips, r.bbox
                FROM regions r
                LEFT JOIN county_infrastructure_summary i USING (county_fips)
                WHERE r.county_fips IS NOT NULL
                  AND r.bbox IS NOT NULL
                  AND i.county_fips IS NULL
            """
        else:
            sql = """
                SELECT r.county_fips, r.bbox
                FROM regions r
                WHERE r.county_fips IS NOT NULL
                  AND r.bbox IS NOT NULL
            """
        rows = await session.execute(text(sql))
        county_rows = rows.fetchall()

    log.info("Counties to process: %d (skip_existing=%s)", len(county_rows), skip_existing)
    if not county_rows:
        log.info("All counties already have infrastructure data.")
        return 0

    sem = asyncio.Semaphore(CONCURRENCY)
    records = []
    errors = 0

    async with httpx.AsyncClient() as client:
        tasks = [_process_county(row.county_fips, row.bbox, client, sem) for row in county_rows]
        results = await asyncio.gather(*tasks)

    for result in results:
        if result is None:
            errors += 1
        else:
            records.append(result)

    log.info(
        "Infrastructure data: %d fetched, %d no usable buildings out of %d",
        len(records),
        errors,
        len(county_rows),
    )

    async with get_async_session() as session:
        for rec in records:
            await session.execute(
                text("""
                INSERT INTO county_infrastructure_summary
                    (county_fips, median_building_age_yr, building_count, updated_at)
                VALUES
                    (:county_fips, :median_building_age_yr, :building_count, now())
                ON CONFLICT (county_fips) DO UPDATE SET
                    median_building_age_yr = EXCLUDED.median_building_age_yr,
                    building_count         = EXCLUDED.building_count,
                    updated_at             = now()
            """),
                rec,
            )

    log.info("Upserted %d county infrastructure summaries", len(records))
    return len(records)


@flow(name="ingest_osm_counties_flow", log_prints=True)
async def ingest_osm_counties_flow(skip_existing: bool = True) -> None:
    """OSM building start_date → median infrastructure age per county.
    Set skip_existing=False to refresh all counties."""
    logger.info("Starting OSM county infrastructure ingestion (skip_existing=%s)", skip_existing)
    try:
        count = await build_county_infrastructure(skip_existing=skip_existing)
        logger.info("OSM infrastructure ingestion complete — %d counties upserted", count)
    except Exception as exc:
        await log_failure("ingest_osm_counties_flow", str(exc))
        raise
