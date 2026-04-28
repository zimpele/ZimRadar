import pytest
from unittest.mock import patch, AsyncMock
from src.ingestion.osm import fetch_osm_buildings, bbox_to_overpass_query


def test_bbox_to_overpass_query_generates_valid_query():
    bbox = {"min_lat": 29.5, "max_lat": 30.1, "min_lon": -95.8, "max_lon": -95.2}
    query = bbox_to_overpass_query(bbox)
    assert "way[building]" in query
    assert "29.5" in query
    assert "-95.8" in query


@pytest.mark.asyncio
async def test_fetch_osm_buildings_returns_geojson():
    mock_response = {
        "elements": [
            {
                "type": "way",
                "id": 123456,
                "tags": {"building": "yes", "start_date": "2005"},
                "nodes": [1, 2, 3, 1],
            }
        ]
    }

    with patch("src.ingestion.osm.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_client
        mock_resp = AsyncMock()
        mock_resp.json.return_value = mock_response
        mock_resp.raise_for_status = AsyncMock()
        mock_client.post.return_value = mock_resp

        bbox = {"min_lat": 29.5, "max_lat": 30.1, "min_lon": -95.8, "max_lon": -95.2}
        result = await fetch_osm_buildings(bbox)

    assert "elements" in result
    assert len(result["elements"]) == 1
