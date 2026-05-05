# src/ingestion/noaa_storm_events.py
"""Prefect flow: ingest NOAA Storm Events CSVs into county_storm_summary."""

import asyncio
import csv
import gzip
import io
import logging
import re
from collections import defaultdict

import httpx
from prefect import flow, task

from src.storage.db import get_async_session
from sqlalchemy import text

logger = logging.getLogger(__name__)

CENSUS_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    "?get=B01003_001E,NAME&for=county:*&in=state:06,08,12,19,37"
)
NOAA_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
YEARS = list(range(2020, 2025))
# CA=06, CO=08, FL=12, IA=19, NC=37
TARGET_STATE_FIPS = {"06", "08", "12", "19", "37"}


def _parse_damage(val: str) -> float:
    if not val or val.strip() in ("", "0"):
        return 0.0
    val = val.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    if val[-1] in multipliers:
        try:
            return float(val[:-1]) * multipliers[val[-1]]
        except ValueError:
            return 0.0
    try:
        return float(val)
    except ValueError:
        return 0.0


@task(name="fetch_census_populations")
async def fetch_census_populations() -> dict[str, int]:
    """Fetch county populations from Census ACS 5-year estimates for CA + FL."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(CENSUS_URL)
        resp.raise_for_status()
        rows = resp.json()

    # First row is headers: ["B01003_001E", "NAME", "state", "county"]
    pop_dict: dict[str, int] = {}
    for row in rows[1:]:
        pop_str, _name, state, county = row
        fips = state + county.zfill(3)
        try:
            pop_dict[fips] = int(pop_str)
        except (ValueError, TypeError):
            pass

    logger.info("Fetched populations for %d counties", len(pop_dict))
    return pop_dict


@task(name="fetch_noaa_index")
async def fetch_noaa_index() -> dict[int, str]:
    """Fetch NOAA index page and return {year: url} for storm events detail files."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(NOAA_INDEX_URL)
        resp.raise_for_status()
        html = resp.text

    year_urls: dict[int, str] = {}
    pattern = re.compile(r'href="(StormEvents_details-ftp_v1\.0_d(\d{4})_[^"]+\.csv\.gz)"')
    for match in pattern.finditer(html):
        filename, year_str = match.group(1), match.group(2)
        year = int(year_str)
        if year in YEARS:
            # Keep last match per year (latest version)
            year_urls[year] = NOAA_INDEX_URL + filename

    logger.info("Found NOAA files for years: %s", sorted(year_urls.keys()))
    return year_urls


def _parse_csv_gz(content: bytes) -> list[dict[str, str]]:
    """Decompress gzip content and parse CSV rows. Run via asyncio.to_thread."""
    with gzip.open(io.BytesIO(content), "rt", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        return list(reader)


@task(name="download_and_parse_year")
async def download_and_parse_year(year: int, url: str) -> list[dict[str, str]]:
    """Download one year's storm events CSV.gz and return parsed rows."""
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content = resp.content

    rows = await asyncio.to_thread(_parse_csv_gz, content)
    logger.info("Year %d: parsed %d rows", year, len(rows))
    return rows


def _aggregate_rows(
    all_rows: list[dict[str, str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Aggregate storm event counts and damage per county FIPS (CA + FL only)."""
    counts: dict[str, int] = defaultdict(int)
    flood_counts: dict[str, int] = defaultdict(int)
    damage: dict[str, float] = defaultdict(float)

    for row in all_rows:
        if row.get("CZ_TYPE") != "C":
            continue
        state_fips = row.get("STATE_FIPS", "").zfill(2)
        cz_fips = row.get("CZ_FIPS", "").zfill(3)
        fips = state_fips + cz_fips
        if fips[:2] not in TARGET_STATE_FIPS:
            continue
        counts[fips] += 1
        if "flood" in row.get("EVENT_TYPE", "").lower():
            flood_counts[fips] += 1
        damage[fips] += _parse_damage(row.get("DAMAGE_PROPERTY", ""))

    return dict(counts), dict(flood_counts), dict(damage)


@task(name="upsert_storm_summary")
async def upsert_storm_summary(
    counts: dict[str, int],
    flood_counts: dict[str, int],
    damage: dict[str, float],
    pop_dict: dict[str, int],
) -> int:
    """Upsert aggregated storm data into county_storm_summary."""
    upsert_sql = text("""
        INSERT INTO county_storm_summary
            (county_fips, storm_events_5yr, flood_events_5yr_noaa,
             storm_damage_usd, population, storm_damage_per_capita, updated_at)
        VALUES
            (:fips, :events, :flood_events, :damage_usd,
             :population, :damage_per_capita, now())
        ON CONFLICT (county_fips) DO UPDATE SET
            storm_events_5yr        = EXCLUDED.storm_events_5yr,
            flood_events_5yr_noaa   = EXCLUDED.flood_events_5yr_noaa,
            storm_damage_usd        = EXCLUDED.storm_damage_usd,
            population              = EXCLUDED.population,
            storm_damage_per_capita = EXCLUDED.storm_damage_per_capita,
            updated_at              = EXCLUDED.updated_at
    """)

    rows_written = 0
    async with get_async_session() as session:
        for fips, event_count in counts.items():
            pop = pop_dict.get(fips, 0)
            dmg = damage.get(fips, 0.0)
            per_capita = dmg / pop if pop > 0 else 0.0
            await session.execute(
                upsert_sql,
                {
                    "fips": fips,
                    "events": event_count,
                    "flood_events": flood_counts.get(fips, 0),
                    "damage_usd": dmg,
                    "population": pop,
                    "damage_per_capita": per_capita,
                },
            )
            rows_written += 1

    logger.info("Upserted %d county storm summaries", rows_written)
    return rows_written


@flow(name="ingest_noaa_storm_events", log_prints=True)
async def ingest_storm_events_flow() -> int:
    """Fetch NOAA Storm Events (2020-2024) for CA + FL and store county summaries."""
    pop_dict = await fetch_census_populations()

    year_urls = await fetch_noaa_index()
    if not year_urls:
        logger.warning("No NOAA files found for years %s", YEARS)
        return 0

    all_rows: list[dict[str, str]] = []
    for year in sorted(year_urls.keys()):
        rows = await download_and_parse_year(year, year_urls[year])
        all_rows.extend(rows)

    logger.info("Total rows across all years: %d", len(all_rows))

    counts, flood_counts, damage = await asyncio.to_thread(_aggregate_rows, all_rows)
    logger.info("Aggregated %d unique counties (CA+FL only)", len(counts))

    rows_written = await upsert_storm_summary(counts, flood_counts, damage, pop_dict)
    return rows_written
