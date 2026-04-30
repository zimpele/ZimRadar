"""Prefect flow: ingest FEMA National Risk Index county scores."""
import csv
import io
import logging
import zipfile
import httpx
from prefect import flow, task, get_run_logger
from sqlalchemy import text
from src.storage.db import get_async_session
from src.ingestion.base import with_retry, log_failure

logger = logging.getLogger(__name__)

NRI_ZIP_URL = (
    "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/"
    "NRI_Table_Counties/NRI_Table_Counties.zip"
)

# NRI CSV column → table column
_COL_MAP = {
    "STCOFIPS":  "county_fips",
    "STATEABBRV": "state_abbr",
    "COUNTY":    "county_name",
    "RISK_SCORE": "risk_score",
    "RISK_RATNG": "risk_rating",
    "EAL_SCORE":  "eal_score",
    "SOVI_SCORE": "sovi_score",
    "RESL_SCORE": "resl_score",
    "CFLD_RISKS": "cfld_risks",
    "RFLD_RISKS": "rfld_risks",
    "HWAV_RISKS": "hwav_risks",
    "DRGT_RISKS": "drgt_risks",
    "WFIR_RISKS": "wfir_risks",
    "SWND_RISKS": "swnd_risks",
    "TRND_RISKS": "trnd_risks",
}


def _float_or_none(val: str) -> float | None:
    try:
        return float(val) if val.strip() else None
    except (ValueError, AttributeError):
        return None


@task(name="download-nri-csv", log_prints=True)
async def download_nri_csv() -> list[dict]:
    log = get_run_logger()
    log.info("Downloading FEMA NRI county CSV from %s", NRI_ZIP_URL)

    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        resp = await client.get(NRI_ZIP_URL)
        resp.raise_for_status()
        raw = resp.content

    log.info("Downloaded %.1f MB, extracting CSV…", len(raw) / 1_048_576)

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV found inside NRI ZIP")
        csv_name = csv_names[0]
        log.info("Extracting %s", csv_name)
        with zf.open(csv_name) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
            rows = list(reader)

    log.info("Parsed %d county rows from NRI CSV", len(rows))
    return rows


@task(name="upsert-nri-counties", log_prints=True)
async def upsert_nri_counties(rows: list[dict]) -> int:
    log = get_run_logger()
    records = []
    for row in rows:
        fips = row.get("STCOFIPS", "").strip()
        if not fips:
            continue
        records.append({
            "county_fips":  fips,
            "state_abbr":   row.get("STATEABBRV", "").strip() or None,
            "county_name":  row.get("COUNTY", "").strip() or None,
            "risk_score":   _float_or_none(row.get("RISK_SCORE", "")),
            "risk_rating":  row.get("RISK_RATNG", "").strip() or None,
            "eal_score":    _float_or_none(row.get("EAL_SCORE", "")),
            "sovi_score":   _float_or_none(row.get("SOVI_SCORE", "")),
            "resl_score":   _float_or_none(row.get("RESL_SCORE", "")),
            "cfld_risks":   _float_or_none(row.get("CFLD_RISKS", "")),
            "rfld_risks":   _float_or_none(row.get("RFLD_RISKS", "")),
            "hwav_risks":   _float_or_none(row.get("HWAV_RISKS", "")),
            "drgt_risks":   _float_or_none(row.get("DRGT_RISKS", "")),
            "wfir_risks":   _float_or_none(row.get("WFIR_RISKS", "")),
            "swnd_risks":   _float_or_none(row.get("SWND_RISKS", "")),
            "trnd_risks":   _float_or_none(row.get("TRND_RISKS", "")),
        })

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
    """Download FEMA National Risk Index CSV and upsert all county scores."""
    logger.info("Starting FEMA NRI ingestion")
    try:
        rows = await with_retry(download_nri_csv, max_attempts=3)
        count = await upsert_nri_counties(rows)
        logger.info("FEMA NRI ingestion complete — %d counties", count)
    except Exception as exc:
        await log_failure("ingest_fema_nri", str(exc))
        raise
