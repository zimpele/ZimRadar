import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.validator import validator_node


@pytest.mark.asyncio
async def test_validator_finalizes_when_score_above_threshold():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    with (
        patch("src.agents.validator.complete", new_callable=AsyncMock, return_value="0.92"),
        patch("src.agents.validator.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {
            "region_id": 1,
            "report_draft": "Climate risk is high [1].",
            "retrieved_context": [{"text": "Flooding is common in this region."}],
            "citations": [
                {
                    "index": 1,
                    "text": "Flooding is common.",
                    "source_type": "fema",
                    "source_id": "123",
                }
            ],
            "risk_tier": "high",
            "risk_score": 0.8,
            "retry_count": 0,
        }
        result = await validator_node(state)

    assert abs(result["factuality_score"] - 0.92) < 0.01
    assert result["final_report"] == "Climate risk is high [1]."
    assert result["low_confidence"] is False
    assert result["report_id"] is not None


@pytest.mark.asyncio
async def test_validator_routes_back_when_score_low_and_retries_remain():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    with (
        patch("src.agents.validator.complete", new_callable=AsyncMock, return_value="0.55"),
        patch("src.agents.validator.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {
            "region_id": 1,
            "report_draft": "Vague report with no citations.",
            "retrieved_context": [{"text": "Some text."}],
            "citations": [],
            "risk_tier": "low",
            "risk_score": 0.3,
            "retry_count": 0,
        }
        result = await validator_node(state)

    assert abs(result["factuality_score"] - 0.55) < 0.01
    assert result["final_report"] is None
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_validator_finalizes_after_max_retries_with_low_confidence():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    with (
        patch("src.agents.validator.complete", new_callable=AsyncMock, return_value="0.55"),
        patch("src.agents.validator.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {
            "region_id": 1,
            "report_draft": "Still vague.",
            "retrieved_context": [],
            "citations": [],
            "risk_tier": "low",
            "risk_score": 0.3,
            "retry_count": 2,
        }
        result = await validator_node(state)

    assert result["final_report"] == "Still vague."
    assert result["low_confidence"] is True
    assert result["report_id"] is not None
