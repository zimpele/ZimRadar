import httpx
import logging
from collections import defaultdict
from datetime import date
from prefect import flow
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure
from src.config import get_settings

logger = logging.getLogger(__name__)
NOAA_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"


async def fetch_noaa_daily(
    station_id: str, start_date: str, end_date: str, api_key: str
) -> list[dict]:
    headers = {"token": api_key}
    params: dict = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "startdate": start_date,
        "enddate": end_date,
        "datatypeid": "PRCP,TMAX,TMIN,AWND",
        "limit": 1000,
        "units": "metric",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(NOAA_BASE, headers=headers, params=params)
        await response.raise_for_status()
        raw = (await response.json()).get("results", [])

    by_date: dict[str, dict] = defaultdict(
        lambda: {
            "precipitation_mm": None,
            "temp_max_c": None,
            "temp_min_c": None,
            "soil_moisture": None,
        }
    )
    for item in raw:
        d = item["date"][:10]
        dt = item["datatype"]
        v = item["value"]
        if dt == "PRCP":
            by_date[d]["precipitation_mm"] = v / 10.0
        elif dt == "TMAX":
            by_date[d]["temp_max_c"] = v / 10.0
        elif dt == "TMIN":
            by_date[d]["temp_min_c"] = v / 10.0

    return [{"date": d, **vals} for d, vals in by_date.items()]


def _parse_date(val: str | date | None) -> date | None:
    if val is None:
        return None
    if isinstance(val, date):
        return val
    try:
        return date.fromisoformat(str(val)[:10])
    except ValueError:
        return None


async def upsert_observations(records: list[dict], session: AsyncSession) -> None:
    for r in records:
        row = dict(r)
        row["date"] = _parse_date(row.get("date"))
        await session.execute(
            text("""
                INSERT INTO noaa_observations
                    (station_id, region_id, date, precipitation_mm, temp_max_c, temp_min_c, soil_moisture)
                VALUES
                    (:station_id, :region_id, :date, :precipitation_mm, :temp_max_c, :temp_min_c, :soil_moisture)
                ON CONFLICT (station_id, date) DO UPDATE SET
                    precipitation_mm = EXCLUDED.precipitation_mm,
                    temp_max_c = EXCLUDED.temp_max_c,
                    temp_min_c = EXCLUDED.temp_min_c,
                    soil_moisture = EXCLUDED.soil_moisture
            """),
            row,
        )


@flow(name="ingest_noaa", log_prints=True)
async def ingest_noaa_flow(region_id: int, station_id: str, start_date: str, end_date: str) -> None:
    settings = get_settings()
    try:
        obs = await with_retry(
            lambda: fetch_noaa_daily(station_id, start_date, end_date, settings.noaa_api_key)
        )
        enriched = [{"station_id": station_id, "region_id": region_id, **o} for o in obs]
        async with get_async_session() as session:
            await upsert_observations(enriched, session)
        logger.info(f"Upserted {len(enriched)} NOAA observations for station {station_id}")
    except Exception as exc:
        await log_failure("ingest_noaa", str(exc), region_id=region_id)
        raise
