"""Prefect flow: batch SegFormer segmentation for all unprocessed tiles."""
import logging

from prefect import flow, task, get_run_logger
from sqlalchemy import text

from src.pipeline.segmentation import run_segmentation_for_tile
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)


@task(name="fetch_unprocessed_tiles", log_prints=True)
async def fetch_unprocessed_tiles(state_codes: list[str]) -> list[dict]:
    """Return tiles that have no segmentation result yet."""
    placeholders = ",".join(f"'{s}'" for s in state_codes)
    sql = text(f"""
        SELECT t.id AS tile_id, t.processed_s3_path
        FROM sentinel2_tiles t
        JOIN regions r ON r.id = t.region_id
        LEFT JOIN segmentation_results sr ON sr.tile_id = t.id
        WHERE r.state_code IN ({placeholders})
          AND t.processed_s3_path IS NOT NULL
          AND sr.tile_id IS NULL
        ORDER BY t.id
    """)
    async with get_async_session() as session:
        rows = (await session.execute(sql)).fetchall()
    return [{"tile_id": row.tile_id, "processed_s3_path": row.processed_s3_path} for row in rows]


@task(name="segment_tile", log_prints=True, retries=1, retry_delay_seconds=10)
async def segment_tile(tile_id: int, processed_s3_path: str) -> bool:
    try:
        await run_segmentation_for_tile(tile_id, processed_s3_path)
        return True
    except Exception as exc:
        logger.warning("Segmentation failed for tile %d: %s", tile_id, exc)
        return False


@flow(name="segment_tiles_flow", log_prints=True, timeout_seconds=14400)
async def segment_tiles_flow(state_codes: list[str] | None = None) -> int:
    """Run SegFormer on all unprocessed tiles for given states.

    Defaults to CA + FL (the fully classified states). Sequential to avoid
    OOM — SegFormer loads a ~400MB model per call.
    """
    log = get_run_logger()
    states = state_codes or ["CA", "FL"]
    tiles = await fetch_unprocessed_tiles(states)
    log.info("Found %d unprocessed tiles for states %s", len(tiles), states)

    if not tiles:
        log.info("All tiles already segmented.")
        return 0

    done = 0
    failed = 0
    for i, tile in enumerate(tiles):
        success = await segment_tile(tile["tile_id"], tile["processed_s3_path"])
        if success:
            done += 1
        else:
            failed += 1
        if (i + 1) % 10 == 0:
            log.info("Progress: %d/%d done, %d failed", done, len(tiles), failed)

    log.info("Segmentation complete: %d done, %d failed out of %d", done, failed, len(tiles))
    return done
