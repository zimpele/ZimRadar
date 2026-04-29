import pytest
from unittest.mock import AsyncMock, patch
from src.agents.report import report_node


@pytest.mark.asyncio
async def test_report_node_sets_draft_and_citations():
    mock_context = [
        {
            "id": 1,
            "text": "Harris County experienced severe flooding in 2023.",
            "source_type": "fema_declaration",
            "source_id": "FEMA-1234",
            "chunk_index": 0,
            "metadata": {"county_fips": "48201"},
            "similarity": 0.92,
            "rerank_score": 1.5,
        }
    ]

    with (
        patch("src.agents.report.retrieve", new_callable=AsyncMock, return_value=mock_context),
        patch(
            "src.agents.report.complete", new_callable=AsyncMock, return_value="Risk is high [1]."
        ),
    ):
        state = {
            "region_query": "Harris County, TX",
            "region_id": 1,
            "risk_tier": "high",
            "risk_score": 0.78,
            "forecast": {"flood_risk_flag": True, "fire_risk_flag": False},
            "retry_count": 0,
        }
        result = await report_node(state)

    assert result["report_draft"] == "Risk is high [1]."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["index"] == 1
    assert result["retrieved_context"] == mock_context


@pytest.mark.asyncio
async def test_report_node_preserves_retry_count():
    with (
        patch("src.agents.report.retrieve", new_callable=AsyncMock, return_value=[]),
        patch("src.agents.report.complete", new_callable=AsyncMock, return_value="Report text."),
    ):
        state = {
            "region_query": "Test Region",
            "region_id": 1,
            "risk_tier": "low",
            "risk_score": 0.2,
            "forecast": {},
            "retry_count": 1,
        }
        result = await report_node(state)

    assert result["report_draft"] == "Report text."
    assert result["retry_count"] == 1
