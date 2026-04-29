import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.analysis import analysis_node


@pytest.mark.asyncio
async def test_analysis_node_sets_risk_tier_and_score():
    mock_session = AsyncMock()
    fc_row = ({"forecast_30d": {}}, True, False)
    ra_row = ("high", 0.87, 0.74)

    mock_session.execute = AsyncMock(
        side_effect=[
            MagicMock(fetchone=MagicMock(return_value=fc_row)),
            MagicMock(fetchone=MagicMock(return_value=ra_row)),
        ]
    )

    with (
        patch("src.agents.analysis.run_forecast_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.run_classification_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_id": 1, "region_query": "Harris County"}
        result = await analysis_node(state)

    assert result["risk_tier"] == "high"
    assert abs(result["risk_score"] - 0.74) < 0.01
    assert "forecast" in result


@pytest.mark.asyncio
async def test_analysis_node_handles_missing_db_rows():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=MagicMock(fetchone=MagicMock(return_value=None)))

    with (
        patch("src.agents.analysis.run_forecast_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.run_classification_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_id": 99, "region_query": "Unknown"}
        result = await analysis_node(state)

    assert result["risk_tier"] == "moderate"
    assert abs(result["risk_score"] - 0.5) < 0.01
