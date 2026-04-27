import numpy as np
from src.pipeline.preprocessing import crop_and_normalize, TILE_SIZE


def test_crop_and_normalize_returns_correct_shape(tmp_path):
    import rasterio
    from rasterio.transform import from_bounds

    big = TILE_SIZE * 2
    data = np.random.randint(0, 3000, (3, big, big), dtype=np.uint16)
    src_path = tmp_path / "big_tile.tif"
    transform = from_bounds(0, 0, 1, 1, big, big)

    with rasterio.open(
        str(src_path),
        "w",
        driver="GTiff",
        height=big,
        width=big,
        count=3,
        dtype="uint16",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    out_path = tmp_path / "cropped.tif"
    crop_and_normalize(str(src_path), str(out_path))

    with rasterio.open(str(out_path)) as src:
        result = src.read()

    assert result.shape == (3, TILE_SIZE, TILE_SIZE)
    assert result.dtype == np.float32
    assert result.max() <= 1.0
    assert result.min() >= 0.0
