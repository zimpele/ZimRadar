import numpy as np
from unittest.mock import patch, MagicMock
from src.pipeline.segmentation import SegmentationPipeline, LAND_USE_CLASSES


def test_land_use_classes_has_five_categories():
    assert set(LAND_USE_CLASSES) == {"water", "vegetation", "urban", "bare_soil", "burn_scar"}


def test_segment_tile_returns_area_stats(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    data = np.random.rand(3, 256, 256).astype(np.float32)
    tile_path = tmp_path / "tile.tif"
    with rasterio.open(
        str(tile_path), "w", driver="GTiff",
        height=256, width=256, count=3, dtype="float32",
        crs="EPSG:4326", transform=from_bounds(0, 0, 1, 1, 256, 256)
    ) as dst:
        dst.write(data)

    with patch("src.pipeline.segmentation.SegformerForSemanticSegmentation.from_pretrained") as mock_model_cls, \
         patch("src.pipeline.segmentation.SegformerImageProcessor.from_pretrained") as mock_proc_cls:

        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_proc = MagicMock()
        mock_proc_cls.return_value = mock_proc

        mock_proc.return_value = {"pixel_values": MagicMock()}
        import torch
        fake_logits = torch.rand(1, 5, 64, 64)
        mock_output = MagicMock()
        mock_output.logits = fake_logits
        mock_model.return_value = mock_output

        fake_seg = torch.zeros(256, 256, dtype=torch.long)
        mock_proc.post_process_semantic_segmentation.return_value = [fake_seg]

        pipeline = SegmentationPipeline()
        result = pipeline.segment(str(tile_path))

    assert "geojson" in result
    assert "area_stats" in result
    assert "flood_zone_geojson" in result
    assert "model_version" in result
    for cls in LAND_USE_CLASSES:
        assert cls in result["area_stats"]
