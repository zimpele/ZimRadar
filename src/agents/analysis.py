from sqlalchemy import text
from src.agents.state import ZimRadarState
from src.pipeline.forecasting import run_forecast_for_region
from src.pipeline.classifier import run_classification_for_region
from src.storage.db import get_async_session


async def analysis_node(state: ZimRadarState) -> ZimRadarState:
    region_id = state.get("region_id", 0)

    await run_forecast_for_region(region_id)
    await run_classification_for_region(region_id)

    async with get_async_session() as session:
        fc_result = await session.execute(
            text("""
                SELECT forecast_30d, flood_risk_flag, fire_risk_flag
                FROM forecasts
                WHERE region_id = :rid
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"rid": region_id},
        )
        fc_row = fc_result.fetchone()

    async with get_async_session() as session:
        ra_result = await session.execute(
            text("""
                SELECT risk_tier, confidence, composite_score
                FROM risk_assessments
                WHERE region_id = :rid
                ORDER BY assessed_at DESC
                LIMIT 1
            """),
            {"rid": region_id},
        )
        ra_row = ra_result.fetchone()

    forecast = {
        "forecast_30d": fc_row[0] if fc_row else {},
        "flood_risk_flag": bool(fc_row[1]) if fc_row else False,
        "fire_risk_flag": bool(fc_row[2]) if fc_row else False,
    }

    return {
        **state,
        "forecast": forecast,
        "risk_tier": ra_row[0] if ra_row else "moderate",
        "risk_score": float(ra_row[2]) if ra_row else 0.5,
    }
