import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.ingest import ingest_node


@pytest.mark.asyncio
async def test_ingest_node_sets_region_id():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=[
        MagicMock(fetchone=MagicMock(return_value=(42,))),
        MagicMock(fetchall=MagicMock(return_value=[("s3://bucket/tile.tif",)])),
    ])

    with patch("src.agents.ingest.get_async_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_query": "Harris County"}
        result = await ingest_node(state)

    assert result["region_id"] == 42


@pytest.mark.asyncio
async def test_ingest_node_region_not_found():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(
        return_value=MagicMock(fetchone=MagicMock(return_value=None))
    )

    with patch("src.agents.ingest.get_async_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_query": "Nonexistent County"}
        result = await ingest_node(state)

    assert "error" in result
    assert "Nonexistent County" in result["error"]
