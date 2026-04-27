import pytest
from sqlalchemy import text
from src.storage.db import get_async_session


@pytest.mark.asyncio
async def test_pgvector_extension_enabled(db_session):
    result = await db_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.scalar() == "vector"


@pytest.mark.asyncio
async def test_all_tables_exist(db_session):
    expected = {
        "regions", "sentinel2_tiles", "noaa_observations", "fema_declarations",
        "segmentation_results", "risk_assessments", "forecasts", "reports",
        "image_embeddings", "text_embeddings", "failed_ingestion",
    }
    result = await db_session.execute(
        text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    )
    tables = {row[0] for row in result.fetchall()}
    assert expected.issubset(tables)
