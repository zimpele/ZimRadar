import asyncio
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from prefect import flow
from rasterio.enums import Resampling as RS
from sentinelsat import SentinelAPI, geojson_to_wkt
from shapely.geometry import box
from sqlalchemy import text

from src.config import get_settings
from src.ingestion.base import log_failure, with_retry
from src.pipeline.preprocessing import REFLECTANCE_MAX, TILE_SIZE
from src.storage.db import get_async_session
from src.storage.s3 import S3Client

logger = logging.getLogger(__name__)


def search_sentinel2_tiles(
    bbox: dict, date_from: str, date_to: str, user: str, password: str
) -> list[dict]:
    api = SentinelAPI(user, password, "https://catalogue.dataspace.copernicus.eu/odata/v1")
    footprint = geojson_to_wkt(
        box(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]).__geo_interface__
    )
    products = api.query(
        footprint,
        date=(date_from, date_to),
        platformname="Sentinel-2",
        cloudcoverpercentage=(0, 20),
    )
    return [{"uuid": uuid, **meta} for uuid, meta in products.items()]


@flow(name="ingest_sentinel2", log_prints=True)
async def ingest_sentinel2_flow(region_id: int, date_from: str, date_to: str) -> None:
    settings = get_settings()
    s3 = S3Client()

    async with get_async_session() as session:
        result = await session.execute(
            text("SELECT bbox FROM regions WHERE id = :id"), {"id": region_id}
        )
        row = result.one_or_none()
        if not row:
            raise ValueError(f"Region {region_id} not found")
        bbox = row[0]

    try:
        products = await with_retry(
            lambda: asyncio.to_thread(
                search_sentinel2_tiles,
                bbox,
                date_from,
                date_to,
                settings.sentinelsat_user,
                settings.sentinelsat_pass,
            )
        )
        logger.info(f"Found {len(products)} Sentinel-2 tiles for region {region_id}")

        api = SentinelAPI(
            settings.sentinelsat_user,
            settings.sentinelsat_pass,
            "https://catalogue.dataspace.copernicus.eu/odata/v1",
        )

        for product in products:
            with tempfile.TemporaryDirectory() as tmpdir:
                await asyncio.to_thread(api.download, product["uuid"], directory_path=tmpdir)
                raw_safe_dirs = list(Path(tmpdir).glob("**/*.SAFE"))
                if not raw_safe_dirs:
                    continue

                safe_dir = raw_safe_dirs[0]
                tile_date = product.get("beginposition", "")[:10] or date_from
                raw_s3_key = await asyncio.to_thread(
                    s3.upload_tile, str(safe_dir), region_id, tile_date
                )

                # Find 10m JP2 bands (Blue B02, Green B03, Red B04) for RGB composite
                jp2_files = sorted(safe_dir.glob("GRANULE/*/IMG_DATA/R10m/*_B0[234]_10m.jp2"))
                if not jp2_files or len(jp2_files) < 3:
                    logger.warning(
                        "Could not find 10m JP2 bands in %s, skipping preprocessing", safe_dir
                    )
                    continue

                band_arrays = []
                profile = None
                for jp2 in jp2_files[:3]:
                    with rasterio.open(str(jp2)) as src:
                        arr = src.read(
                            1,
                            out_shape=(TILE_SIZE, TILE_SIZE),
                            resampling=RS.bilinear,
                        ).astype(np.float32)
                        if profile is None:
                            profile = src.profile.copy()
                            scale_x = src.width / TILE_SIZE
                            scale_y = src.height / TILE_SIZE
                            scaled_transform = src.transform * src.transform.scale(scale_x, scale_y)
                    band_arrays.append(arr)

                stacked = np.stack(band_arrays)
                stacked = np.clip(stacked, 0, REFLECTANCE_MAX) / REFLECTANCE_MAX

                processed_path = os.path.join(tmpdir, f"processed_{product['uuid']}.tif")
                profile.update(
                    driver="GTiff",
                    count=3,
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    dtype="float32",
                    transform=scaled_transform,
                )
                with rasterio.open(processed_path, "w", **profile) as dst:
                    dst.write(stacked)

                processed_s3_key = await asyncio.to_thread(
                    s3.upload_processed_tile, processed_path, region_id, tile_date
                )

                async with get_async_session() as session:
                    await session.execute(
                        text("""
                            INSERT INTO sentinel2_tiles
                                (region_id, s3_path, processed_s3_path, date, ingested_at)
                            VALUES (:region_id, :s3_path, :processed_s3_path, :date, NOW())
                            ON CONFLICT (region_id, s3_path) DO UPDATE
                                SET processed_s3_path = EXCLUDED.processed_s3_path
                        """),
                        {
                            "region_id": region_id,
                            "s3_path": raw_s3_key,
                            "processed_s3_path": processed_s3_key,
                            "date": tile_date,
                        },
                    )

    except Exception as exc:
        await log_failure("ingest_sentinel2", str(exc), region_id=region_id)
        raise
