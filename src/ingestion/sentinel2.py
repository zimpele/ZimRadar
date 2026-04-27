import logging
import tempfile
import os
from pathlib import Path
from sentinelsat import SentinelAPI, geojson_to_wkt
from shapely.geometry import box
from prefect import flow
from sqlalchemy import text
from src.storage.db import get_async_session
from src.storage.s3 import S3Client
from src.pipeline.preprocessing import crop_and_normalize
from src.ingestion.base import with_retry, log_failure
from src.config import get_settings

logger = logging.getLogger(__name__)


def search_sentinel2_tiles(
    bbox: dict, date_from: str, date_to: str, user: str, password: str
) -> list[dict]:
    api = SentinelAPI(user, password, "https://scihub.copernicus.eu/dhus")
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
            lambda: search_sentinel2_tiles(
                bbox, date_from, date_to,
                settings.sentinelsat_user, settings.sentinelsat_pass,
            )
        )
        logger.info(f"Found {len(products)} Sentinel-2 tiles for region {region_id}")

        api = SentinelAPI(
            settings.sentinelsat_user, settings.sentinelsat_pass,
            "https://scihub.copernicus.eu/dhus",
        )

        for product in products:
            with tempfile.TemporaryDirectory() as tmpdir:
                api.download(product["uuid"], directory_path=tmpdir)
                raw_files = list(Path(tmpdir).glob("**/*.SAFE"))
                if not raw_files:
                    continue

                tile_date = product.get("beginposition", "")[:10] or date_from
                raw_s3_key = s3.upload_tile(str(raw_files[0]), region_id, tile_date)

                processed_path = os.path.join(tmpdir, f"processed_{product['uuid']}.tif")
                crop_and_normalize(str(raw_files[0]), processed_path)
                processed_s3_key = s3.upload_processed_tile(processed_path, region_id, tile_date)

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
