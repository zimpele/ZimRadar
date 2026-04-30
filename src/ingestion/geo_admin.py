"""Fetch and store administrative boundary geometries for regions."""
import asyncio
import json
import logging
from typing import Any

import httpx
from sqlalchemy import text

from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

CENSUS_COUNTIES_GEOJSON = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# US state FIPS prefix → 2-letter state code
FIPS_TO_STATE: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}

# One high-risk county per US state — chosen for FEMA declaration frequency and
# climate exposure (coastal flooding, wildfire, extreme heat, hurricane tracks).
ALL_50_COUNTIES: dict[str, str] = {
    "01097": "Mobile County, AL",
    "02170": "Matanuska-Susitna Borough, AK",
    "04013": "Maricopa County, AZ",
    "05119": "Pulaski County, AR",
    "06037": "Los Angeles County, CA",
    "08031": "Denver County, CO",
    "09003": "Hartford County, CT",
    "10003": "New Castle County, DE",
    "12086": "Miami-Dade County, FL",
    "13121": "Fulton County, GA",
    "15003": "Honolulu County, HI",
    "16001": "Ada County, ID",
    "17031": "Cook County, IL",
    "18097": "Marion County, IN",
    "19153": "Polk County, IA",
    "20173": "Sedgwick County, KS",
    "21111": "Jefferson County, KY",
    "22071": "Orleans Parish, LA",
    "23005": "Cumberland County, ME",
    "24005": "Baltimore County, MD",
    "25025": "Suffolk County, MA",
    "26163": "Wayne County, MI",
    "27053": "Hennepin County, MN",
    "28049": "Hinds County, MS",
    "29095": "Jackson County, MO",
    "30013": "Cascade County, MT",
    "31055": "Douglas County, NE",
    "32003": "Clark County, NV",
    "33011": "Hillsborough County, NH",
    "34017": "Hudson County, NJ",
    "35001": "Bernalillo County, NM",
    "36061": "New York County, NY",
    "37119": "Mecklenburg County, NC",
    "38017": "Cass County, ND",
    "39035": "Cuyahoga County, OH",
    "40143": "Tulsa County, OK",
    "41051": "Multnomah County, OR",
    "42101": "Philadelphia County, PA",
    "44007": "Providence County, RI",
    "45019": "Charleston County, SC",
    "46099": "Minnehaha County, SD",
    "47157": "Shelby County, TN",
    "48201": "Harris County, TX",
    "49035": "Salt Lake County, UT",
    "50007": "Chittenden County, VT",
    "51710": "Norfolk city, VA",
    "53033": "King County, WA",
    "54039": "Kanawha County, WV",
    "55079": "Milwaukee County, WI",
    "56021": "Laramie County, WY",
}

_county_features: dict[str, dict] | None = None


async def _get_county_features() -> dict[str, dict]:
    """Download and cache US counties GeoJSON indexed by 5-digit FIPS.

    Each entry: {"geometry": {...}, "name": "County Name"}.
    """
    global _county_features
    if _county_features is not None:
        return _county_features
    logger.info("Downloading US counties GeoJSON (~11 MB)…")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(CENSUS_COUNTIES_GEOJSON)
        resp.raise_for_status()
        data = resp.json()
    _county_features = {
        f["id"]: {
            "geometry": f["geometry"],
            "name": f["properties"].get("name", ""),
        }
        for f in data["features"]
    }
    logger.info("Loaded %d county geometries", len(_county_features))
    return _county_features


def _bbox_from_geojson(geometry: dict) -> dict:
    coords: list[list[float]] = []
    if geometry["type"] == "Polygon":
        for ring in geometry["coordinates"]:
            coords.extend(ring)
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            for ring in poly:
                coords.extend(ring)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {
        "min_lon": min(lons),
        "min_lat": min(lats),
        "max_lon": max(lons),
        "max_lat": max(lats),
    }


async def add_county_region(fips: str, display_name: str | None = None) -> int:
    """Insert a US county region by FIPS code and return its region id.

    If a region with this FIPS already exists its geometry is back-filled if
    missing, and the existing id is returned without creating a duplicate.
    """
    state_code = FIPS_TO_STATE.get(fips[:2])
    if not state_code:
        raise ValueError(f"Unknown state FIPS prefix: {fips[:2]}")

    features = await _get_county_features()
    entry = features.get(fips)
    if not entry:
        raise ValueError(f"FIPS {fips} not found in Census data")

    name = display_name or f"{entry['name']}, {state_code}"
    geometry = entry["geometry"]
    bbox = _bbox_from_geojson(geometry)

    async with get_async_session() as session:
        existing = await session.execute(
            text("SELECT id, geometry FROM regions WHERE county_fips = :fips"),
            {"fips": fips},
        )
        row = existing.fetchone()
        if row:
            if row[1] is None:
                await session.execute(
                    text("""
                        UPDATE regions
                        SET geometry   = CAST(:geometry AS jsonb),
                            bbox       = CAST(:bbox AS jsonb),
                            state_code = :state_code
                        WHERE id = :rid
                    """),
                    {
                        "rid": row[0],
                        "geometry": json.dumps(geometry),
                        "bbox": json.dumps(bbox),
                        "state_code": state_code,
                    },
                )
                logger.info("Back-filled geometry for region %d (FIPS %s)", row[0], fips)
            return row[0]

        result = await session.execute(
            text("""
                INSERT INTO regions (name, bbox, state_code, county_fips, geometry, active)
                VALUES (:name, CAST(:bbox AS jsonb), :state_code, :fips,
                        CAST(:geometry AS jsonb), true)
                RETURNING id
            """),
            {
                "name": name,
                "bbox": json.dumps(bbox),
                "state_code": state_code,
                "fips": fips,
                "geometry": json.dumps(geometry),
            },
        )
        region_id = result.fetchone()[0]
        logger.info("Created region %d: %s (FIPS %s)", region_id, name, fips)
        return region_id


async def list_counties_for_state(state_code: str) -> list[tuple[str, str]]:
    """Return [(county_name, fips), ...] for a US state, sorted by county name."""
    state_fips = next(
        (k for k, v in FIPS_TO_STATE.items() if v == state_code), None
    )
    if not state_fips:
        return []
    features = await _get_county_features()
    return sorted(
        [
            (entry["name"], fips)
            for fips, entry in features.items()
            if fips[:2] == state_fips
        ],
        key=lambda x: x[0],
    )


async def _seed_bw_region() -> None:
    """Populate geometry for the Baden-Württemberg region (expects id=1)."""
    BW_BBOX = {"min_lon": 7.5, "min_lat": 47.5, "max_lon": 10.5, "max_lat": 49.8}
    geom = None
    try:
        async with httpx.AsyncClient(
            timeout=30.0, headers={"User-Agent": "ZimRadar/1.0"}
        ) as client:
            resp = await client.get(
                NOMINATIM_URL,
                params={
                    "q": "Baden-Württemberg, Germany",
                    "format": "geojson",
                    "polygon_geojson": "1",
                    "limit": "1",
                },
            )
            resp.raise_for_status()
            features = resp.json().get("features", [])
            if features:
                geom = features[0]["geometry"]
    except Exception as exc:
        logger.warning("BW Nominatim fetch failed (%s), using bbox fallback", exc)

    if not geom:
        geom = {
            "type": "Polygon",
            "coordinates": [[
                [BW_BBOX["min_lon"], BW_BBOX["min_lat"]],
                [BW_BBOX["max_lon"], BW_BBOX["min_lat"]],
                [BW_BBOX["max_lon"], BW_BBOX["max_lat"]],
                [BW_BBOX["min_lon"], BW_BBOX["max_lat"]],
                [BW_BBOX["min_lon"], BW_BBOX["min_lat"]],
            ]],
        }

    bbox = _bbox_from_geojson(geom)
    async with get_async_session() as session:
        await session.execute(
            text("""
                UPDATE regions
                SET state_code = 'DE-BW',
                    geometry   = CAST(:geometry AS jsonb),
                    bbox       = CAST(:bbox AS jsonb)
                WHERE id = 1
            """),
            {"geometry": json.dumps(geom), "bbox": json.dumps(bbox)},
        )
    logger.info("Updated BW geometry (region id=1)")


async def run_all() -> None:
    """Seed all 50 representative US counties and populate BW geometry."""
    async with get_async_session() as session:
        await session.execute(text("""
            ALTER TABLE regions
                ADD COLUMN IF NOT EXISTS state_code  TEXT,
                ADD COLUMN IF NOT EXISTS county_fips TEXT,
                ADD COLUMN IF NOT EXISTS geometry    JSONB
        """))

    for fips, display_name in ALL_50_COUNTIES.items():
        try:
            await add_county_region(fips, display_name)
        except Exception as exc:
            logger.error("Failed to seed FIPS %s: %s", fips, exc)

    await _seed_bw_region()
    logger.info("Geometry population complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_all())
