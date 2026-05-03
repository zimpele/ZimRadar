"""Prefect flow: ingest FEMA National Risk Index county scores via ArcGIS FeatureServer."""

import logging
import httpx
from prefect import flow, task, get_run_logger
from sqlalchemy import text
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure

logger = logging.getLogger(__name__)

NRI_FEATURE_SERVER = (
    "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services"
    "/National_Risk_Index_Counties/FeatureServer/0/query"
)
PAGE_SIZE = 2000
NRI_FIELDS = ",".join(
    [
        "STCOFIPS",
        "STATEABBRV",
        "COUNTY",
        "RISK_SCORE",
        "RISK_RATNG",
        "EAL_SCORE",
        "SOVI_SCORE",
        "RESL_SCORE",
        "CFLD_RISKS",
        "IFLD_RISKS",
        "HWAV_RISKS",
        "DRGT_RISKS",
        "WFIR_RISKS",
        "SWND_RISKS",
        "TRND_RISKS",
    ]
)


def _f(attrs: dict, key: str) -> float | None:
    v = attrs.get(key)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


@task(name="download-nri-features", log_prints=True)
async def download_nri_features() -> list[dict]:
    log = get_run_logger()
    log.info("Fetching FEMA NRI counties from ArcGIS FeatureServer…")

    all_features = []
    offset = 0
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            resp = await client.post(
                NRI_FEATURE_SERVER,
                data={
                    "where": "1=1",
                    "outFields": NRI_FIELDS,
                    "returnGeometry": "false",
                    "resultOffset": str(offset),
                    "resultRecordCount": str(PAGE_SIZE),
                    "f": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"ArcGIS error: {data['error']}")
            batch = data.get("features", [])
            all_features.extend(batch)
            log.info(
                "  page offset=%d → %d features (total so far: %d)",
                offset,
                len(batch),
                len(all_features),
            )
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    log.info("Downloaded %d NRI county features", len(all_features))
    return all_features


@task(name="upsert-nri-counties", log_prints=True)
async def upsert_nri_counties(features: list[dict]) -> int:
    log = get_run_logger()
    records = []
    for feat in features:
        attrs = feat.get("attributes", {})
        fips = (attrs.get("STCOFIPS") or "").strip()
        if not fips:
            continue
        records.append(
            {
                "county_fips": fips,
                "state_abbr": (attrs.get("STATEABBRV") or "").strip() or None,
                "county_name": (attrs.get("COUNTY") or "").strip() or None,
                "risk_score": _f(attrs, "RISK_SCORE"),
                "risk_rating": (attrs.get("RISK_RATNG") or "").strip() or None,
                "eal_score": _f(attrs, "EAL_SCORE"),
                "sovi_score": _f(attrs, "SOVI_SCORE"),
                "resl_score": _f(attrs, "RESL_SCORE"),
                "cfld_risks": _f(attrs, "CFLD_RISKS"),
                "rfld_risks": _f(attrs, "IFLD_RISKS"),  # inland flood = riverine flood
                "hwav_risks": _f(attrs, "HWAV_RISKS"),
                "drgt_risks": _f(attrs, "DRGT_RISKS"),
                "wfir_risks": _f(attrs, "WFIR_RISKS"),
                "swnd_risks": _f(attrs, "SWND_RISKS"),
                "trnd_risks": _f(attrs, "TRND_RISKS"),
            }
        )

    async with get_async_session() as session:
        for rec in records:
            await session.execute(
                text("""
                    INSERT INTO fema_nri_county
                        (county_fips, state_abbr, county_name,
                         risk_score, risk_rating, eal_score, sovi_score, resl_score,
                         cfld_risks, rfld_risks, hwav_risks, drgt_risks,
                         wfir_risks, swnd_risks, trnd_risks, updated_at)
                    VALUES
                        (:county_fips, :state_abbr, :county_name,
                         :risk_score, :risk_rating, :eal_score, :sovi_score, :resl_score,
                         :cfld_risks, :rfld_risks, :hwav_risks, :drgt_risks,
                         :wfir_risks, :swnd_risks, :trnd_risks, now())
                    ON CONFLICT (county_fips) DO UPDATE SET
                        state_abbr   = EXCLUDED.state_abbr,
                        county_name  = EXCLUDED.county_name,
                        risk_score   = EXCLUDED.risk_score,
                        risk_rating  = EXCLUDED.risk_rating,
                        eal_score    = EXCLUDED.eal_score,
                        sovi_score   = EXCLUDED.sovi_score,
                        resl_score   = EXCLUDED.resl_score,
                        cfld_risks   = EXCLUDED.cfld_risks,
                        rfld_risks   = EXCLUDED.rfld_risks,
                        hwav_risks   = EXCLUDED.hwav_risks,
                        drgt_risks   = EXCLUDED.drgt_risks,
                        wfir_risks   = EXCLUDED.wfir_risks,
                        swnd_risks   = EXCLUDED.swnd_risks,
                        trnd_risks   = EXCLUDED.trnd_risks,
                        updated_at   = now()
                """),
                rec,
            )

    log.info("Upserted %d NRI county records", len(records))
    return len(records)


@flow(name="ingest_fema_nri", log_prints=True)
async def ingest_fema_nri_flow() -> None:
    """Fetch FEMA National Risk Index from ArcGIS FeatureServer and upsert all county scores."""
    logger.info("Starting FEMA NRI ingestion")
    try:
        features = await with_retry(download_nri_features, max_attempts=3)
        count = await upsert_nri_counties(features)
        logger.info("FEMA NRI ingestion complete — %d counties", count)
    except Exception as exc:
        await log_failure("ingest_fema_nri", str(exc))
        raise
