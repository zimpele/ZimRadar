import pytest
from unittest.mock import patch, AsyncMock
from sqlalchemy import text
from src.ingestion.fema import fetch_fema_declarations, upsert_declarations


@pytest.mark.asyncio
async def test_fetch_fema_declarations_returns_list():
    mock_response = {
        "DisasterDeclarationsSummaries": [
            {
                "disasterNumber": "4332",
                "state": "TX",
                "designatedArea": "Harris",
                "fipsCountyCode": "201",
                "incidentType": "Flood",
                "declarationDate": "2017-09-07T00:00:00.000Z",
                "incidentBeginDate": "2017-08-25T00:00:00.000Z",
                "incidentEndDate": "2017-09-15T00:00:00.000Z",
                "declarationTitle": "HURRICANE HARVEY",
            }
        ]
    }

    with patch("src.ingestion.fema.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_response_obj = AsyncMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status = AsyncMock()
        mock_client.get.return_value = mock_response_obj

        results = await fetch_fema_declarations(last_refresh=None)

    assert len(results) == 1
    assert results[0]["disasterNumber"] == "4332"


@pytest.mark.asyncio
async def test_upsert_declarations_is_idempotent(db_session):
    record = {
        "disaster_number": "DR-4332",
        "state": "TX",
        "county_fips": "48201",
        "disaster_type": "Flood",
        "declaration_date": "2017-09-07",
        "incident_begin": "2017-08-25",
        "incident_end": "2017-09-15",
        "declaration_title": "HURRICANE HARVEY",
    }

    await upsert_declarations([record], db_session)
    await upsert_declarations([record], db_session)  # second call must not raise

    result = await db_session.execute(
        text("SELECT COUNT(*) FROM fema_declarations WHERE disaster_number = 'DR-4332'")
    )
    assert result.scalar() == 1
