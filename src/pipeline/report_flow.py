"""Prefect flow: generate AI risk reports for all classified counties."""

import asyncio
import json
import logging

from prefect import flow, get_run_logger, task
from sqlalchemy import text

from src.agents.report_agent import generate_county_report
from src.ingestion.base import log_failure
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)
CONCURRENCY = 1  # free OpenRouter tier has low RPM; sequential avoids 429s


@task(name="generate-report-for-county", log_prints=True)
async def generate_report_task(
    region_id: int,
    county_fips: str,
    county_name: str,
    risk_tier: str,
    confidence: float,
    shap_json: str,
    features_json: str,
) -> bool:
    log = get_run_logger()
    try:
        shap_dict = json.loads(shap_json) if isinstance(shap_json, str) else shap_json
        features = json.loads(features_json) if isinstance(features_json, str) else features_json
        result = await generate_county_report(
            region_id=region_id,
            county_fips=county_fips,
            county_name=county_name,
            risk_tier=risk_tier,
            confidence=confidence,
            shap_dict=shap_dict or {},
            features=features or {},
        )
        flagged = result.get("flagged", False)
        log.info("Region %d (%s): report generated (flagged=%s)", region_id, county_name, flagged)
        return not flagged
    except Exception as exc:
        log.error("Region %d failed: %s", region_id, exc)
        return False


@flow(name="generate_county_reports", log_prints=True)
async def generate_county_reports_flow(
    state_codes: list[str] | None = None,
    skip_existing: bool = True,
) -> int:
    """Generate AI risk briefings for all classified counties using LangGraph agent."""
    log = get_run_logger()

    async with get_async_session() as session:
        base_sql = """
            SELECT ra.region_id, r.county_fips, r.name, r.state_code,
                   ra.risk_tier, ra.confidence, ra.shap_values, ra.features_json
            FROM risk_assessments ra
            JOIN regions r ON r.id = ra.region_id
            WHERE ra.shap_values IS NOT NULL
              AND r.county_fips IS NOT NULL
              AND ra.assessed_at = (
                  SELECT MAX(ra2.assessed_at) FROM risk_assessments ra2
                  WHERE ra2.region_id = ra.region_id
              )
        """
        params: dict = {}
        if state_codes:
            base_sql += " AND r.state_code = ANY(:states)"
            params["states"] = state_codes
        if skip_existing:
            base_sql += (
                " AND NOT EXISTS"
                " (SELECT 1 FROM county_reports cr WHERE cr.region_id = ra.region_id)"
            )
        base_sql += " ORDER BY ra.composite_score DESC"

        result = await session.execute(text(base_sql), params)
        rows = result.fetchall()

    if not rows:
        log.info("No counties need reports — all done or none classified.")
        return 0

    log.info("Generating reports for %d counties (concurrency=%d)", len(rows), CONCURRENCY)
    sem = asyncio.Semaphore(CONCURRENCY)
    ok = 0
    errors = 0

    async def run_one(row) -> None:
        nonlocal ok, errors
        async with sem:
            success = await generate_report_task(
                region_id=row.region_id,
                county_fips=row.county_fips,
                county_name=f"{row.name}, {row.state_code}",
                risk_tier=row.risk_tier,
                confidence=float(row.confidence or 0),
                shap_json=row.shap_values,
                features_json=row.features_json,
            )
            if success:
                ok += 1
            else:
                errors += 1

    try:
        await asyncio.gather(*[run_one(row) for row in rows])
    except Exception as exc:
        await log_failure("generate_county_reports", str(exc))
        raise

    log.info("Reports complete: %d ok, %d errors", ok, errors)
    return ok
