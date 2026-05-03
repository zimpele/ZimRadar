"""Prefect flow: bulk NOAA station lookup + precipitation for all FEMA counties."""

import asyncio
import logging
import numpy as np
import httpx
from datetime import date, timedelta
from prefect import flow, task, get_run_logger
from sqlalchemy import text
from src.storage.db import get_async_session
from src.config import get_settings
from src.ingestion.base import log_failure

logger = logging.getLogger(__name__)

NOAA_STATIONS_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
NOAA_DATA_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
CONCURRENCY = 3  # conservative — share across station + data calls
REQ_DELAY = 0.35  # seconds between requests per slot → ~3 req/s per slot
FETCH_DAYS = 730  # 2 years of daily data


def _date_range() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=FETCH_DAYS)
    return start.isoformat(), end.isoformat()


async def _get(
    url: str,
    params: dict,
    api_key: str,
    client: httpx.AsyncClient,
    rate_sem: asyncio.Semaphore,
) -> dict:
    """Rate-limited GET with retry on 429."""
    for attempt in range(3):
        async with rate_sem:
            try:
                resp = await client.get(
                    url,
                    headers={"token": api_key},
                    params=params,
                    timeout=30.0,
                )
                if resp.status_code == 429:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                await asyncio.sleep(REQ_DELAY)
                return resp.json()
            except httpx.TimeoutException:
                await asyncio.sleep(1.0)
                continue
            except Exception:
                return {}
    return {}


async def _best_station(
    fips: str,
    api_key: str,
    client: httpx.AsyncClient,
    rate_sem: asyncio.Semaphore,
) -> str | None:
    data = await _get(
        NOAA_STATIONS_URL,
        {
            "locationid": f"FIPS:{fips}",
            "datasetid": "GHCND",
            "datatypeid": "PRCP",
            "limit": 10,
            "sortfield": "maxdate",
            "sortorder": "desc",
        },
        api_key,
        client,
        rate_sem,
    )
    results = data.get("results", [])
    # prefer stations active in last 2 years
    cutoff = (date.today() - timedelta(days=730)).isoformat()
    recent = [r for r in results if r.get("maxdate", "") >= cutoff]
    chosen = recent or results
    return chosen[0]["id"] if chosen else None


async def _fetch_precip(
    station_id: str,
    start: str,
    end: str,
    api_key: str,
    client: httpx.AsyncClient,
    rate_sem: asyncio.Semaphore,
) -> list[float]:
    data = await _get(
        NOAA_DATA_URL,
        {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": "PRCP",
            "startdate": start,
            "enddate": end,
            "units": "metric",
            "limit": 1000,
        },
        api_key,
        client,
        rate_sem,
    )
    return [r["value"] / 10.0 for r in data.get("results", []) if r.get("value") is not None]


def _compute_trend(values: list[float]) -> tuple[float, float]:
    if len(values) < 14:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float32)
    x = np.arange(len(arr), dtype=np.float32)
    return float(np.mean(arr)), float(np.polyfit(x, arr, 1)[0])


@task(name="build-county-climate", log_prints=True)
async def build_county_climate(skip_existing: bool = True) -> int:
    log = get_run_logger()
    settings = get_settings()
    api_key = settings.noaa_api_key
    if not api_key:
        raise RuntimeError("NOAA_API_KEY is not set")

    async with get_async_session() as session:
        if skip_existing:
            sql = """
                SELECT DISTINCT f.county_fips
                FROM fema_declarations f
                LEFT JOIN county_climate_summary c USING (county_fips)
                WHERE f.county_fips IS NOT NULL
                  AND length(f.county_fips) = 5
                  AND RIGHT(f.county_fips, 3) != '000'
                  AND c.county_fips IS NULL
            """
        else:
            sql = """
                SELECT DISTINCT county_fips
                FROM fema_declarations
                WHERE county_fips IS NOT NULL
                  AND length(county_fips) = 5
                  AND RIGHT(county_fips, 3) != '000'
            """
        rows = await session.execute(text(sql))
        fips_list = [r[0] for r in rows.fetchall()]

    log.info("Counties to process: %d (skip_existing=%s)", len(fips_list), skip_existing)
    if not fips_list:
        log.info("All counties already have climate data.")
        return 0

    start, end = _date_range()
    rate_sem = asyncio.Semaphore(CONCURRENCY)
    records = []
    errors = 0

    async def process_county(fips: str, client: httpx.AsyncClient) -> dict | None:
        nonlocal errors
        station_id = await _best_station(fips, api_key, client, rate_sem)
        if not station_id:
            errors += 1
            return None
        values = await _fetch_precip(station_id, start, end, api_key, client, rate_sem)
        avg, trend = _compute_trend(values)
        return {
            "county_fips": fips,
            "station_id": station_id,
            "avg_precip_mm": avg,
            "precip_trend": trend,
            "obs_days": len(values),
        }

    async with httpx.AsyncClient() as client:
        tasks = [process_county(fips, client) for fips in fips_list]
        results = await asyncio.gather(*tasks)

    records = [r for r in results if r is not None]
    log.info(
        "Climate data: %d fetched, %d no station found out of %d",
        len(records),
        errors,
        len(fips_list),
    )

    async with get_async_session() as session:
        for rec in records:
            await session.execute(
                text("""
                INSERT INTO county_climate_summary
                    (county_fips, station_id, avg_precip_mm, precip_trend, obs_days, updated_at)
                VALUES
                    (:county_fips, :station_id, :avg_precip_mm, :precip_trend, :obs_days, now())
                ON CONFLICT (county_fips) DO UPDATE SET
                    station_id    = EXCLUDED.station_id,
                    avg_precip_mm = EXCLUDED.avg_precip_mm,
                    precip_trend  = EXCLUDED.precip_trend,
                    obs_days      = EXCLUDED.obs_days,
                    updated_at    = now()
            """),
                rec,
            )

    log.info("Upserted %d county climate summaries", len(records))
    return len(records)


@flow(name="ingest_noaa_counties", log_prints=True)
async def ingest_noaa_counties_flow(skip_existing: bool = True) -> None:
    """Find best NOAA station per FEMA county and fetch 2yr precipitation trend.
    Set skip_existing=False to refresh all counties."""
    logger.info("Starting bulk NOAA county climate ingestion (skip_existing=%s)", skip_existing)
    try:
        count = await build_county_climate(skip_existing=skip_existing)
        logger.info("NOAA county ingestion complete — %d counties upserted", count)
    except Exception as exc:
        await log_failure("ingest_noaa_counties", str(exc))
        raise
