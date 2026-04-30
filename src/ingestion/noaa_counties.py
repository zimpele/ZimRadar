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
NOAA_DATA_URL     = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
CONCURRENCY       = 5   # NOAA CDO rate limit: 5 req/s
FETCH_DAYS        = 730  # 2 years of daily data


def _date_range() -> tuple[str, str]:
    end   = date.today()
    start = end - timedelta(days=FETCH_DAYS)
    return start.isoformat(), end.isoformat()


async def _best_station(fips: str, api_key: str, client: httpx.AsyncClient) -> str | None:
    """Return station_id of the best GHCND precipitation station in this county."""
    try:
        resp = await client.get(
            NOAA_STATIONS_URL,
            headers={"token": api_key},
            params={
                "locationid": f"FIPS:{fips}",
                "datasetid": "GHCND",
                "datatypeid": "PRCP",
                "limit": 10,
                "sortfield": "maxdate",
                "sortorder": "desc",
            },
            timeout=20.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0]["id"] if results else None
    except Exception:
        return None


async def _fetch_precip(
    station_id: str, start: str, end: str, api_key: str, client: httpx.AsyncClient
) -> list[float]:
    """Fetch daily PRCP values (mm) for a station over the date window."""
    try:
        resp = await client.get(
            NOAA_DATA_URL,
            headers={"token": api_key},
            params={
                "datasetid":  "GHCND",
                "stationid":  station_id,
                "datatypeid": "PRCP",
                "startdate":  start,
                "enddate":    end,
                "units":      "metric",
                "limit":      1000,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return [r["value"] / 10.0 for r in resp.json().get("results", []) if r.get("value") is not None]
    except Exception:
        return []


def _compute_trend(values: list[float]) -> tuple[float, float]:
    """Return (avg_mm, slope_mm_per_day) or (0, 0) if insufficient data."""
    if len(values) < 14:
        return 0.0, 0.0
    arr = np.array(values, dtype=np.float32)
    avg = float(np.mean(arr))
    x = np.arange(len(arr), dtype=np.float32)
    slope = float(np.polyfit(x, arr, 1)[0])
    return avg, slope


@task(name="build-county-climate", log_prints=True)
async def build_county_climate() -> int:
    log = get_run_logger()
    settings = get_settings()
    api_key = settings.noaa_api_key
    if not api_key:
        raise RuntimeError("NOAA_API_KEY is not set")

    async with get_async_session() as session:
        rows = await session.execute(text(
            "SELECT DISTINCT county_fips FROM fema_declarations "
            "WHERE county_fips IS NOT NULL AND county_fips != ''"
        ))
        fips_list = [r[0] for r in rows.fetchall()]

    log.info("Found %d distinct county FIPS codes", len(fips_list))
    start, end = _date_range()
    sem = asyncio.Semaphore(CONCURRENCY)
    upserted = 0

    async def process_county(fips: str, client: httpx.AsyncClient) -> dict | None:
        async with sem:
            station_id = await _best_station(fips, api_key, client)
            if not station_id:
                return None
            await asyncio.sleep(0.2)  # stay within 5 req/s
            values = await _fetch_precip(station_id, start, end, api_key, client)
            await asyncio.sleep(0.2)
        avg, trend = _compute_trend(values)
        return {
            "county_fips":   fips,
            "station_id":    station_id,
            "avg_precip_mm": avg,
            "precip_trend":  trend,
            "obs_days":      len(values),
        }

    async with httpx.AsyncClient() as client:
        tasks = [process_county(fips, client) for fips in fips_list]
        results = await asyncio.gather(*tasks)

    records = [r for r in results if r is not None]
    log.info("Got climate data for %d / %d counties", len(records), len(fips_list))

    async with get_async_session() as session:
        for rec in records:
            await session.execute(text("""
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
            """), rec)
        upserted = len(records)

    log.info("Upserted %d county climate summaries", upserted)
    return upserted


@flow(name="ingest_noaa_counties", log_prints=True)
async def ingest_noaa_counties_flow() -> None:
    """Find best NOAA station per FEMA county and fetch 2yr precipitation trend."""
    logger.info("Starting bulk NOAA county climate ingestion")
    try:
        count = await build_county_climate()
        logger.info("NOAA county ingestion complete — %d counties", count)
    except Exception as exc:
        await log_failure("ingest_noaa_counties", str(exc))
        raise
