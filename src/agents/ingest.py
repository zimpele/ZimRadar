from sqlalchemy import text
from src.agents.state import ZimRadarState
from src.storage.db import get_async_session


async def ingest_node(state: ZimRadarState) -> ZimRadarState:
    region_query = state.get("region_query", "")

    async with get_async_session() as session:
        result = await session.execute(
            text("SELECT id FROM regions WHERE name ILIKE :q LIMIT 1"),
            {"q": f"%{region_query}%"},
        )
        row = result.fetchone()

    if row is None:
        return {**state, "error": f"Region '{region_query}' not found in database"}

    region_id: int = row[0]

    async with get_async_session() as session:
        tiles = await session.execute(
            text("""
                SELECT s3_path FROM sentinel2_tiles
                WHERE region_id = :rid
                ORDER BY date DESC
                LIMIT 5
            """),
            {"rid": region_id},
        )
        tile_paths = [r[0] for r in tiles.fetchall()]

    return {
        **state,
        "region_id": region_id,
        "tile_paths": tile_paths,
        "segmentation_results": {},
        "depth_map": {},
    }
