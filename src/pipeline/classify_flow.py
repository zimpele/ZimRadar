"""Prefect flow: batch classify all active regions and store risk assessments."""

import asyncio
import logging

from prefect import flow, task, get_run_logger
from sqlalchemy import text

from src.pipeline.classifier import run_classification_for_region
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

CONCURRENCY = 2


@task(name="fetch_regions_to_classify", log_prints=True)
async def fetch_regions_to_classify(
    state_codes: list[str] | None,
    skip_existing: bool,
) -> list[int]:
    """Return region IDs to classify."""
    if state_codes:
        placeholders = ",".join(f"'{s}'" for s in state_codes)
        state_filter = f"AND r.state_code IN ({placeholders})"
    else:
        state_filter = ""

    if skip_existing:
        sql = text(f"""
            SELECT r.id FROM regions r
            LEFT JOIN LATERAL (
                SELECT 1 FROM risk_assessments ra
                WHERE ra.region_id = r.id
                ORDER BY ra.assessed_at DESC LIMIT 1
            ) latest ON TRUE
            WHERE r.active = TRUE
              AND r.county_fips IS NOT NULL
              AND latest IS NULL
              {state_filter}
            ORDER BY r.id
        """)
    else:
        sql = text(f"""
            SELECT id FROM regions
            WHERE active = TRUE AND county_fips IS NOT NULL
            {state_filter}
            ORDER BY id
        """)

    async with get_async_session() as session:
        rows = (await session.execute(sql)).fetchall()
    return [row[0] for row in rows]


@flow(name="classify_regions_flow", log_prints=True, timeout_seconds=7200)
async def classify_regions_flow(
    state_codes: list[str] | None = None,
    skip_existing: bool = False,
) -> int:
    """Classify all active regions and store results to risk_assessments.

    Args:
        state_codes: Limit to specific states (e.g. ['CA','FL']). None = all.
        skip_existing: If True, only classify regions with no existing assessment.
    """
    log = get_run_logger()
    region_ids = await fetch_regions_to_classify(state_codes, skip_existing)
    log.info("Regions to classify: %d", len(region_ids))

    if not region_ids:
        log.info("Nothing to do.")
        return 0

    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    failed = 0

    async def _classify(rid: int) -> bool:
        async with sem:
            try:
                await run_classification_for_region(rid)
                return True
            except Exception as exc:
                logger.warning("Classification failed for region %d: %s", rid, exc)
                return False

    results = await asyncio.gather(*[_classify(rid) for rid in region_ids])
    done = sum(results)
    failed = len(results) - done

    log.info("Classification complete: %d done, %d failed out of %d", done, failed, len(region_ids))
    return done
