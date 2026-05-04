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
CONCURRENCY = 1
RETRY_DELAYS = (10, 30, 60)  # seconds to wait after 429, then give up
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


async def _overpass_post(query: str, client: httpx.AsyncClient, sem: asyncio.Semaphore) -> dict:
    async with sem:
        for attempt, delay in enumerate((*RETRY_DELAYS, None)):
            try:
                resp = await client.post(OVERPASS_URL, data={"data": query}, timeout=60.0)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and delay is not None:
                    logger.warning(
                        "Overpass 429 — retrying in %ds (attempt %d)", delay, attempt + 1
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning("Overpass request failed: %s", exc)
                    return {}
            except Exception as exc:
                logger.warning("Overpass request failed: %s", exc)
                return {}
        return {}


async def _process_county(
    county_fips: str,
    bbox: dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Return infrastructure summary dict, or None on total fetch failure.

    Strategy:
    1. Fast count query → total building count (always available).
    2. Small age-tag query → buildings with explicit construction year (rare in US).
    If age tags exist: store median age (years).
    Otherwise: store building-count-derived development score as proxy.
    """
    min_lat = float(bbox["min_lat"])
    max_lat = float(bbox["max_lat"])
    min_lon = float(bbox["min_lon"])
    max_lon = float(bbox["max_lon"])
    ob = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    # 1. Count all buildings (fast — no element download)
    count_q = (
        f'[out:json][timeout:30];(way["building"]({ob});relation["building"]({ob}););out count;'
    )
    count_data = await _overpass_post(count_q, client, sem)
    elements = count_data.get("elements", [])
    if not elements:
        return None
    building_count = int(elements[0].get("tags", {}).get("total", 0))

    # 2. Query buildings with any known age tag (tiny result set)
    age_union = "".join(
        f'way["building"]["{t}"]({ob});relation["building"]["{t}"]({ob});' for t in AGE_TAGS
    )
    age_q = f"[out:json][timeout:30];({age_union});out tags;"
    age_data = await _overpass_post(age_q, client, sem)
    years = []
    for el in age_data.get("elements", []):
        tags = el.get("tags", {})
        for tag in AGE_TAGS:
            year = _parse_year(tags.get(tag, ""))
            if year is not None:
                years.append(year)
                break

    if years:
        age = float(CURRENT_YEAR - int(median(years)))
    else:
        age = _development_score(building_count)

    return {
        "county_fips": county_fips,
        "median_building_age_yr": age,
        "building_count": building_count,
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
    total_upserted = 0
    total_errors = 0
    BATCH_SIZE = 30
    upsert_sql = text("""
        INSERT INTO county_infrastructure_summary
            (county_fips, median_building_age_yr, building_count, updated_at)
        VALUES
            (:county_fips, :median_building_age_yr, :building_count, now())
        ON CONFLICT (county_fips) DO UPDATE SET
            median_building_age_yr = EXCLUDED.median_building_age_yr,
            building_count         = EXCLUDED.building_count,
            updated_at             = now()
    """)

    async with httpx.AsyncClient(headers={"User-Agent": "ZimRadar/1.0"}) as client:
        for batch_start in range(0, len(county_rows), BATCH_SIZE):
            batch = county_rows[batch_start : batch_start + BATCH_SIZE]
            tasks = [_process_county(row.county_fips, row.bbox, client, sem) for row in batch]
            results = await asyncio.gather(*tasks)

            records = [r for r in results if r is not None]
            total_errors += sum(1 for r in results if r is None)

            if records:
                async with get_async_session() as session:
                    for rec in records:
                        await session.execute(upsert_sql, rec)

                total_upserted += len(records)
                log.info(
                    "Batch %d-%d: upserted %d, running total %d",
                    batch_start,
                    batch_start + len(batch) - 1,
                    len(records),
                    total_upserted,
                )

    log.info(
        "Infrastructure data: %d fetched, %d no usable buildings out of %d",
        total_upserted,
        total_errors,
        len(county_rows),
    )
    return total_upserted


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
