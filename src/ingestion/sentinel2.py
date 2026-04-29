import asyncio
import logging
import os
import tempfile
from datetime import date as date_
from pathlib import Path

import httpx
import numpy as np
import rasterio
from prefect import flow
from pystac_client import Client as STACClient
from rasterio.enums import Resampling as RS
from sqlalchemy import text

from src.config import get_settings
from src.ingestion.base import log_failure, with_retry
from src.pipeline.preprocessing import REFLECTANCE_MAX, TILE_SIZE
from src.storage.db import get_async_session
from src.storage.s3 import S3Client

logger = logging.getLogger(__name__)

STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"


def _get_token(username: str, password: str) -> str:
    resp = httpx.post(
        TOKEN_URL,
        data={
            "grant_type": "password",
            "username": username,
            "password": password,
            "client_id": "cdse-public",
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _get_product_uuid(item_id: str, token: str) -> str | None:
    """Look up OData UUID by SAFE product name (requires Bearer token)."""
    safe_name = item_id + ".SAFE"
    resp = httpx.get(
        f"{ODATA_BASE}/Products",
        params={
            "$filter": f"Name eq '{safe_name}'",
            "$select": "Id,Name",
            "$top": "1",
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=30.0,
    )
    resp.raise_for_status()
    values = resp.json().get("value", [])
    if not values:
        logger.warning("OData: product not found for %s", safe_name)
        return None
    return values[0]["Id"]


def _s3_to_nodes_url(s3_uri: str, product_uuid: str) -> str:
    """
    Convert an S3 URI from a STAC asset to a Nodes download URL.

    s3://eodata/.../S2A_MSIL2A_....SAFE/GRANULE/.../R10m/TCI_10m.jp2
    →
    https://catalogue.dataspace.copernicus.eu/odata/v1/Products(UUID)
        /Nodes(S2A_MSIL2A_....SAFE)/Nodes(GRANULE)/.../$value
    """
    path = s3_uri.removeprefix("s3://eodata/")
    safe_split = path.split(".SAFE/", 1)
    safe_name = safe_split[0].rsplit("/", 1)[-1] + ".SAFE"
    inner_path = safe_split[1]  # GRANULE/granule_id/IMG_DATA/R10m/filename.jp2

    parts = [safe_name] + inner_path.split("/")
    nodes = "/".join(f"Nodes({p})" for p in parts)
    return f"{ODATA_BASE}/Products({product_uuid})/{nodes}/$value"


def _search_tiles(bbox: dict, date_from: str, date_to: str, max_items: int = 3) -> list:
    catalog = STACClient.open(STAC_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        datetime=f"{date_from}/{date_to}",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=max_items,
        sortby="-datetime",
    )
    return list(search.items())


def _download_file(url: str, token: str, dest_path: str) -> None:
    """Download via OData Nodes, following redirects manually."""
    session = httpx.Client(timeout=300.0)
    headers = {"Authorization": f"Bearer {token}"}
    # Follow redirects manually — OData may redirect to the actual blob URL
    response = session.get(url, headers=headers, follow_redirects=False)
    while response.status_code in (301, 302, 303, 307, 308):
        redirect_url = response.headers["Location"]
        response = session.get(redirect_url, headers=headers, follow_redirects=False)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)


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
        items = await asyncio.to_thread(_search_tiles, bbox, date_from, date_to)
        logger.info(f"Found {len(items)} Sentinel-2 L2A tiles for region {region_id}")

        if not items:
            logger.warning(f"No tiles found for region {region_id} in {date_from}–{date_to}")
            return

        token = await asyncio.to_thread(
            _get_token, settings.sentinelsat_user, settings.sentinelsat_pass
        )

        for item in items:
            tile_date_str = item.datetime.strftime("%Y-%m-%d") if item.datetime else date_from
            tile_date = date_.fromisoformat(tile_date_str)
            assets = item.assets

            # Prefer TCI (True Colour Image) — already an RGB composite at 10m
            # Fall back to individual B02/B03/B04 bands
            if "TCI_10m" in assets:
                tci_s3 = assets["TCI_10m"].href
                use_tci = True
            elif all(k in assets for k in ("B04_10m", "B03_10m", "B02_10m")):
                band_s3s = [
                    assets["B04_10m"].href,
                    assets["B03_10m"].href,
                    assets["B02_10m"].href,
                ]
                use_tci = False
            else:
                logger.warning(f"Item {item.id}: no usable RGB assets, skipping")
                continue

            # Resolve OData UUID for this product
            product_uuid = await asyncio.to_thread(_get_product_uuid, item.id, token)
            if product_uuid is None:
                logger.warning(f"Item {item.id}: UUID not found in OData, skipping")
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                if use_tci:
                    tci_path = os.path.join(tmpdir, "tci.jp2")
                    tci_url = _s3_to_nodes_url(tci_s3, product_uuid)
                    logger.info(f"Downloading TCI: {tci_url}")
                    await asyncio.to_thread(_download_file, tci_url, token, tci_path)
                    with rasterio.open(tci_path) as src:
                        stacked = src.read(
                            out_shape=(3, TILE_SIZE, TILE_SIZE),
                            resampling=RS.bilinear,
                        ).astype(np.float32)
                        profile = src.profile.copy()
                        scale_x = src.width / TILE_SIZE
                        scale_y = src.height / TILE_SIZE
                        scaled_transform = src.transform * src.transform.scale(scale_x, scale_y)
                    stacked = np.clip(stacked, 0, 255) / 255.0
                else:
                    band_arrays = []
                    profile = None
                    for i, s3_uri in enumerate(band_s3s):
                        dest = os.path.join(tmpdir, f"band_{i}.jp2")
                        band_url = _s3_to_nodes_url(s3_uri, product_uuid)
                        await asyncio.to_thread(_download_file, band_url, token, dest)
                        with rasterio.open(dest) as src:
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
                    stacked = np.clip(np.stack(band_arrays), 0, REFLECTANCE_MAX) / REFLECTANCE_MAX

                processed_path = os.path.join(tmpdir, f"processed_{item.id}.tif")
                profile.update(
                    driver="GTiff",
                    count=3,
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    dtype="float32",
                    transform=scaled_transform,
                    crs="EPSG:32632",
                )
                with rasterio.open(processed_path, "w", **profile) as dst:
                    dst.write(stacked)

                raw_s3_key = s3.upload_tile(processed_path, region_id, tile_date)
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
                logger.info(f"Ingested tile {item.id} date={tile_date_str} for region {region_id}")

    except Exception as exc:
        await log_failure("ingest_sentinel2", str(exc), region_id=region_id)
        raise
