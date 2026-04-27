import httpx
import logging
from datetime import datetime, date
from prefect import flow
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure

logger = logging.getLogger(__name__)

FEMA_BASE_URL = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
PAGE_SIZE = 1000


async def fetch_fema_declarations(last_refresh: str | None) -> list[dict]:
    params: dict = {"$top": PAGE_SIZE, "$orderby": "lastRefresh asc"}
    if last_refresh:
        params["$filter"] = f"lastRefresh gt '{last_refresh}'"

    records = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        skip = 0
        while True:
            params["$skip"] = skip
            response = await client.get(FEMA_BASE_URL, params=params)
            await response.raise_for_status()
            data = await response.json()
            batch = data.get("DisasterDeclarationsSummaries", [])
            records.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            skip += PAGE_SIZE

    return records


def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
    except ValueError:
        return None


async def upsert_declarations(records: list[dict], session: AsyncSession) -> None:
    for r in records:
        row = dict(r)
        row["declaration_date"] = _parse_date(row.get("declaration_date"))
        row["incident_begin"] = _parse_date(row.get("incident_begin"))
        row["incident_end"] = _parse_date(row.get("incident_end"))
        await session.execute(
            text("""
                INSERT INTO fema_declarations
                    (disaster_number, state, county_fips, disaster_type,
                     declaration_date, incident_begin, incident_end, declaration_title)
                VALUES
                    (:disaster_number, :state, :county_fips, :disaster_type,
                     :declaration_date, :incident_begin, :incident_end, :declaration_title)
                ON CONFLICT (disaster_number) DO UPDATE SET
                    state = EXCLUDED.state,
                    county_fips = EXCLUDED.county_fips,
                    disaster_type = EXCLUDED.disaster_type,
                    declaration_date = EXCLUDED.declaration_date,
                    incident_begin = EXCLUDED.incident_begin,
                    incident_end = EXCLUDED.incident_end,
                    declaration_title = EXCLUDED.declaration_title
            """),
            row,
        )


@flow(name="ingest_fema", log_prints=True)
async def ingest_fema_flow(last_refresh: str | None = None) -> None:
    logger.info("Starting FEMA ingestion")
    try:
        records = await with_retry(
            lambda: fetch_fema_declarations(last_refresh), max_attempts=3
        )
        logger.info(f"Fetched {len(records)} FEMA records")

        normalized = [
            {
                "disaster_number": r.get("disasterNumber", ""),
                "state": r.get("state"),
                "county_fips": r.get("fipsCountyCode"),
                "disaster_type": r.get("incidentType"),
                "declaration_date": r.get("declarationDate"),
                "incident_begin": r.get("incidentBeginDate"),
                "incident_end": r.get("incidentEndDate"),
                "declaration_title": r.get("declarationTitle"),
            }
            for r in records
            if r.get("disasterNumber")
        ]

        async with get_async_session() as session:
            await upsert_declarations(normalized, session)

        logger.info(f"Upserted {len(normalized)} declarations")
    except Exception as exc:
        await log_failure("ingest_fema", str(exc))
        raise
