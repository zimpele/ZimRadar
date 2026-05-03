from unittest.mock import MagicMock, patch

from src.ingestion.sentinel2 import _search_tiles


def test_search_returns_item_list():
    mock_item = MagicMock()
    mock_item.id = "S2A_MSIL2A_20240115T000000"

    mock_search = MagicMock()
    mock_search.items.return_value = [mock_item]

    mock_catalog = MagicMock()
    mock_catalog.search.return_value = mock_search

    with patch("src.ingestion.sentinel2.STACClient") as mock_stac_cls:
        mock_stac_cls.open.return_value = mock_catalog

        bbox = {"min_lat": 29.5, "max_lat": 30.1, "min_lon": -95.8, "max_lon": -95.2}
        items = _search_tiles(bbox=bbox, date_from="2024-01-10", date_to="2024-01-20")

    assert len(items) == 1
    assert items[0].id == "S2A_MSIL2A_20240115T000000"
    mock_catalog.search.assert_called_once_with(
        collections=["sentinel-2-l2a"],
        bbox=[-95.8, 29.5, -95.2, 30.1],
        datetime="2024-01-10/2024-01-20",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=3,
        sortby="-datetime",
    )
