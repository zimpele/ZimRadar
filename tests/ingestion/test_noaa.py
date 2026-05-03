import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy import text
from src.ingestion.noaa import fetch_noaa_daily, upsert_observations


@pytest.mark.asyncio
async def test_fetch_noaa_daily_returns_observations():
    mock_response = {
        "results": [
            {
                "date": "2024-01-15T00:00:00",
                "station": "GHCND:USC00410613",
                "datatype": "PRCP",
                "value": 25,
            },
            {
                "date": "2024-01-15T00:00:00",
                "station": "GHCND:USC00410613",
                "datatype": "TMAX",
                "value": 180,
            },
        ]
    }

    with patch("src.ingestion.noaa.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_resp = AsyncMock()
        mock_resp.json = MagicMock(return_value=mock_response)
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp

        results = await fetch_noaa_daily(
            station_id="GHCND:USC00410613",
            start_date="2024-01-15",
            end_date="2024-01-15",
            api_key="test-key",
        )

    assert len(results) == 1
    assert results[0]["precipitation_mm"] == pytest.approx(2.5)  # PRCP in tenths of mm
    assert results[0]["temp_max_c"] == pytest.approx(18.0)  # TMAX in tenths of °C


@pytest.mark.asyncio
async def test_upsert_observations_is_idempotent(db_session):
    # Need a region row to satisfy FK
    await db_session.execute(
        text(
            "INSERT INTO regions (name, bbox) VALUES ('test_noaa', '{\"min_lon\": 0}') ON CONFLICT DO NOTHING"
        )
    )
    await db_session.flush()
    region_id = (
        await db_session.execute(text("SELECT id FROM regions WHERE name='test_noaa'"))
    ).scalar()

    obs = {
        "station_id": "GHCND:USC00410613",
        "region_id": region_id,
        "date": "2024-01-15",
        "precipitation_mm": 2.5,
        "temp_max_c": 18.0,
        "temp_min_c": 10.0,
        "soil_moisture": None,
    }

    await upsert_observations([obs], db_session)
    await upsert_observations([obs], db_session)

    result = await db_session.execute(
        text("SELECT COUNT(*) FROM noaa_observations WHERE station_id='GHCND:USC00410613'")
    )
    assert result.scalar() == 1
