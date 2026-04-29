import httpx
import logging
from prefect import flow
from sqlalchemy import text
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure

logger = logging.getLogger(__name__)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def bbox_to_overpass_query(bbox: dict) -> str:
    s, n = bbox["min_lat"], bbox["max_lat"]
    w, e = bbox["min_lon"], bbox["max_lon"]
    return f"""
    [out:json][timeout:60];
    (
      way[building]({s},{w},{n},{e});
      way[highway]({s},{w},{n},{e});
    );
    out body;
    """


async def fetch_osm_buildings(bbox: dict) -> dict:
    query = bbox_to_overpass_query(bbox)
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(OVERPASS_URL, data={"data": query})
        await response.raise_for_status()
        return await response.json()


@flow(name="ingest_osm", log_prints=True)
async def ingest_osm_flow(region_id: int) -> None:
    async with get_async_session() as session:
        result = await session.execute(
            text("SELECT bbox FROM regions WHERE id = :id"), {"id": region_id}
        )
        row = result.one_or_none()
        if not row:
            raise ValueError(f"Region {region_id} not found")
        bbox = row[0]

    try:
        geojson = await with_retry(lambda: fetch_osm_buildings(bbox))
        async with get_async_session() as session:
            await session.execute(
                text("""
                    UPDATE regions SET bbox = jsonb_set(bbox, '{osm_snapshot}', CAST(:snapshot AS jsonb))
                    WHERE id = :id
                """),
                {"snapshot": str(geojson), "id": region_id},
            )
        logger.info(
            f"OSM snapshot updated for region {region_id}: {len(geojson.get('elements', []))} elements"
        )
    except Exception as exc:
        await log_failure("ingest_osm", str(exc), region_id=region_id)
        raise
