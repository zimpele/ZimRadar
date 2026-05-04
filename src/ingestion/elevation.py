"""Prefect flow: SRTM elevation stats per county via USGS EPQS (5x5 grid sample)."""

import asyncio
import logging
import numpy as np
import httpx
from prefect import flow, task, get_run_logger
from sqlalchemy import text
from src.storage.db import get_async_session
from src.ingestion.base import log_failure

logger = logging.getLogger(__name__)

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
GRID_SIZE = 5  # 5x5 = 25 sample points per county
CONCURRENCY = 8
NO_DATA_SENTINEL = -1_000_000.0


async def _fetch_elevation(
    lat: float,
    lon: float,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> float | None:
    """Return elevation in meters for a single point, or None on error / no-data."""
    params = {
        "x": lon,
        "y": lat,
        "wkid": 4326,
        "units": "Meters",
        "includeDate": "false",
    }
    async with sem:
        try:
            resp = await client.get(EPQS_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            value = resp.json().get("value")
            if value is None or float(value) <= NO_DATA_SENTINEL:
                return None
            return float(value)
        except Exception:
            return None


async def _sample_county(
    county_fips: str,
    bbox: dict,
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> dict | None:
    """Sample a GRID_SIZE x GRID_SIZE grid within bbox and compute mean/std."""
    min_lat = float(bbox["min_lat"])
    max_lat = float(bbox["max_lat"])
    min_lon = float(bbox["min_lon"])
    max_lon = float(bbox["max_lon"])

    lats = np.linspace(min_lat, max_lat, GRID_SIZE)
    lons = np.linspace(min_lon, max_lon, GRID_SIZE)

    tasks = [_fetch_elevation(float(lat), float(lon), client, sem) for lat in lats for lon in lons]
    results = await asyncio.gather(*tasks)
    values = [v for v in results if v is not None]

    if not values:
        return None

    arr = np.array(values, dtype=np.float64)
    return {
        "county_fips": county_fips,
        "elevation_mean_m": float(np.mean(arr)),
        "elevation_std_m": float(np.std(arr)),
    }


@task(name="build-county-elevation", log_prints=True)
async def build_county_elevation(skip_existing: bool = True) -> int:
    log = get_run_logger()

    async with get_async_session() as session:
        if skip_existing:
            sql = """
                SELECT r.county_fips, r.bbox
                FROM regions r
                LEFT JOIN county_elevation_summary e USING (county_fips)
                WHERE r.county_fips IS NOT NULL
                  AND r.bbox IS NOT NULL
                  AND e.county_fips IS NULL
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
        log.info("All counties already have elevation data.")
        return 0

    sem = asyncio.Semaphore(CONCURRENCY)
    records = []
    errors = 0

    async with httpx.AsyncClient() as client:
        tasks = [_sample_county(row.county_fips, row.bbox, client, sem) for row in county_rows]
        results = await asyncio.gather(*tasks)

    for result in results:
        if result is None:
            errors += 1
        else:
            records.append(result)

    log.info(
        "Elevation data: %d fetched, %d failed out of %d",
        len(records),
        errors,
        len(county_rows),
    )

    async with get_async_session() as session:
        for rec in records:
            await session.execute(
                text("""
                INSERT INTO county_elevation_summary
                    (county_fips, elevation_mean_m, elevation_std_m, updated_at)
                VALUES
                    (:county_fips, :elevation_mean_m, :elevation_std_m, now())
                ON CONFLICT (county_fips) DO UPDATE SET
                    elevation_mean_m = EXCLUDED.elevation_mean_m,
                    elevation_std_m  = EXCLUDED.elevation_std_m,
                    updated_at       = now()
            """),
                rec,
            )

    log.info("Upserted %d county elevation summaries", len(records))
    return len(records)


@flow(name="ingest_elevation_flow", log_prints=True)
async def ingest_elevation_flow(skip_existing: bool = True) -> None:
    """Sample USGS SRTM elevation grid per county and compute std.
    Set skip_existing=False to refresh all counties."""
    logger.info("Starting county elevation ingestion (skip_existing=%s)", skip_existing)
    try:
        count = await build_county_elevation(skip_existing=skip_existing)
        logger.info("Elevation ingestion complete — %d counties upserted", count)
    except Exception as exc:
        await log_failure("ingest_elevation_flow", str(exc))
        raise
