"""Database query tools for the county report agent."""

from sqlalchemy import text
from src.storage.db import get_async_session


async def fetch_fema_flood_history(county_fips: str) -> dict:
    """FEMA flood/hurricane declarations for the county (last 10 years)."""
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT disaster_type, COUNT(*) as count,
                       MAX(declaration_date) as most_recent
                FROM fema_declarations
                WHERE county_fips = :fips
                  AND disaster_type IN ('Flood','Hurricane','Severe Storm','Tropical Storm','Coastal Storm')
                  AND declaration_date >= NOW() - INTERVAL '10 years'
                GROUP BY disaster_type
                ORDER BY count DESC
            """),
            {"fips": county_fips},
        )
        rows = result.fetchall()
    return {
        "source": "FEMA Declarations",
        "county_fips": county_fips,
        "flood_declarations": [
            {"type": r.disaster_type, "count": r.count, "most_recent": str(r.most_recent)[:10]}
            for r in rows
        ],
        "total": sum(r.count for r in rows),
    }


async def fetch_nri_comparison(county_fips: str) -> dict:
    """Compare county NRI scores to state averages."""
    async with get_async_session() as session:
        county_result = await session.execute(
            text("""
                SELECT risk_score, eal_score, sovi_score, cfld_risks, rfld_risks,
                       wfir_risks, hwav_risks, state_abbr, county_name
                FROM fema_nri_county WHERE county_fips = :fips
            """),
            {"fips": county_fips},
        )
        county = county_result.fetchone()
        if not county:
            return {"source": "FEMA NRI", "available": False}

        state_result = await session.execute(
            text("""
                SELECT AVG(risk_score) as avg_risk, AVG(wfir_risks) as avg_fire,
                       AVG(cfld_risks + rfld_risks) / 2 as avg_flood,
                       AVG(hwav_risks) as avg_heat
                FROM fema_nri_county WHERE state_abbr = :state
            """),
            {"state": county.state_abbr},
        )
        state = state_result.fetchone()

    return {
        "source": "FEMA National Risk Index",
        "county": county.county_name,
        "state": county.state_abbr,
        "county_risk_score": float(county.risk_score or 0),
        "state_avg_risk_score": round(float(state.avg_risk or 0), 1),
        "county_fire_risk": float(county.wfir_risks or 0),
        "state_avg_fire_risk": round(float(state.avg_fire or 0), 1),
        "county_flood_risk": max(float(county.cfld_risks or 0), float(county.rfld_risks or 0)),
        "state_avg_flood_risk": round(float(state.avg_flood or 0), 1),
        "county_heat_risk": float(county.hwav_risks or 0),
        "state_avg_heat_risk": round(float(state.avg_heat or 0), 1),
    }


async def fetch_storm_events_summary(county_fips: str) -> dict:
    """NOAA Storm Events summary (2020-2024)."""
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT storm_events_5yr, flood_events_5yr_noaa,
                       storm_damage_usd, storm_damage_per_capita, population
                FROM county_storm_summary WHERE county_fips = :fips
            """),
            {"fips": county_fips},
        )
        row = result.fetchone()
    if not row:
        return {"source": "NOAA Storm Events", "available": False}
    return {
        "source": "NOAA Storm Events (2020-2024)",
        "total_events": row.storm_events_5yr,
        "flood_events": row.flood_events_5yr_noaa,
        "total_damage_usd": row.storm_damage_usd,
        "damage_per_capita_usd": round(row.storm_damage_per_capita, 2),
        "population": row.population,
    }


async def fetch_climate_trend(county_fips: str) -> dict:
    """NOAA precipitation trend (2-year)."""
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT avg_precip_mm, precip_trend, obs_days, station_id
                FROM county_climate_summary WHERE county_fips = :fips
            """),
            {"fips": county_fips},
        )
        row = result.fetchone()
    if not row:
        return {"source": "NOAA Climate Summary", "available": False}
    trend_dir = (
        "increasing"
        if (row.precip_trend or 0) > 0.01
        else "decreasing"
        if (row.precip_trend or 0) < -0.01
        else "stable"
    )
    return {
        "source": "NOAA CDO (2-year daily)",
        "avg_precipitation_mm_day": round(float(row.avg_precip_mm or 0), 2),
        "precipitation_trend": round(float(row.precip_trend or 0), 4),
        "trend_direction": trend_dir,
        "observation_days": row.obs_days,
        "station_id": row.station_id,
    }


async def fetch_infrastructure_summary(county_fips: str) -> dict:
    """OSM building age and elevation data."""
    async with get_async_session() as session:
        infra = await session.execute(
            text(
                "SELECT median_building_age_yr FROM county_infrastructure_summary"
                " WHERE county_fips = :fips"
            ),
            {"fips": county_fips},
        )
        infra_row = infra.fetchone()

        elev = await session.execute(
            text("SELECT elevation_std_m FROM county_elevation_summary WHERE county_fips = :fips"),
            {"fips": county_fips},
        )
        elev_row = elev.fetchone()

    return {
        "source": "OSM / USGS",
        "median_building_age_years": float(infra_row.median_building_age_yr) if infra_row else None,
        "elevation_std_m": round(float(elev_row.elevation_std_m), 1) if elev_row else None,
    }
