from unittest.mock import patch, MagicMock
from src.ingestion.sentinel2 import search_sentinel2_tiles


def test_search_returns_product_list():
    with patch("src.ingestion.sentinel2.SentinelAPI") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.query.return_value = {
            "abc123": {"title": "S2A_tile_20240115", "size": "500 MB", "uuid": "abc123"}
        }

        bbox = {"min_lat": 29.5, "max_lat": 30.1, "min_lon": -95.8, "max_lon": -95.2}
        products = search_sentinel2_tiles(
            bbox=bbox,
            date_from="2024-01-10",
            date_to="2024-01-20",
            user="user",
            password="pass",
        )

    assert len(products) == 1
    assert products[0]["uuid"] == "abc123"
