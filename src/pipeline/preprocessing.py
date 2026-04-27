import numpy as np
import rasterio
from rasterio.enums import Resampling

TILE_SIZE = 256
REFLECTANCE_MAX = 3000.0  # Sentinel-2 typical surface reflectance max


def crop_and_normalize(src_path: str, dst_path: str) -> None:
    with rasterio.open(src_path) as src:
        data = src.read(
            [1, 2, 3],
            out_shape=(3, TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear,
        ).astype(np.float32)

        data = np.clip(data, 0, REFLECTANCE_MAX) / REFLECTANCE_MAX

        profile = src.profile.copy()
        profile.update(
            count=3,
            height=TILE_SIZE,
            width=TILE_SIZE,
            dtype="float32",
            transform=src.transform,
        )

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)
