import asyncio
import streamlit as st
from sqlalchemy import text
from src.storage.db import get_async_session

st.set_page_config(page_title="ZimRadar", layout="wide", page_icon="🌍")
st.title("🌍 ZimRadar — Climate Risk Assessment")


async def get_regions() -> list[dict]:
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT r.id, r.name, r.bbox,
                       ra.risk_tier, ra.composite_score, ra.assessed_at
                FROM regions r
                LEFT JOIN LATERAL (
                    SELECT risk_tier, composite_score, assessed_at
                    FROM risk_assessments
                    WHERE region_id = r.id
                    ORDER BY assessed_at DESC LIMIT 1
                ) ra ON TRUE
                WHERE r.active = TRUE
                ORDER BY r.name
            """)
        )
        return [dict(row._mapping) for row in result.fetchall()]


async def get_report(region_id: int) -> dict | None:
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT narrative, citations, factuality_score, low_confidence, created_at
                FROM reports
                WHERE region_id = :region_id
                ORDER BY created_at DESC LIMIT 1
            """),
            {"region_id": region_id},
        )
        row = result.one_or_none()
        return dict(row._mapping) if row else None


TIER_COLORS = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}

col_map, col_report = st.columns([3, 2])

with col_map:
    st.subheader("Tracked Regions")
    regions = asyncio.run(get_regions())

    if not regions:
        st.info("No regions tracked yet. Add a region to the `regions` table to get started.")
    else:
        import leafmap.foliumap as leafmap

        m = leafmap.Map(center=[37.5, -96], zoom=4)

        for region in regions:
            bbox = region.get("bbox") or {}
            tier = region.get("risk_tier", "unknown")
            color = {
                "critical": "red",
                "high": "orange",
                "moderate": "yellow",
                "low": "green",
            }.get(tier, "gray")
            if all(k in bbox for k in ("min_lon", "min_lat", "max_lon", "max_lat")):
                m.add_geojson(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [bbox["min_lon"], bbox["min_lat"]],
                                [bbox["max_lon"], bbox["min_lat"]],
                                [bbox["max_lon"], bbox["max_lat"]],
                                [bbox["min_lon"], bbox["max_lat"]],
                                [bbox["min_lon"], bbox["min_lat"]],
                            ]],
                        },
                        "properties": {"name": region["name"], "tier": tier},
                    },
                    layer_name=region["name"],
                    style={"color": color, "fillOpacity": 0.2},
                )
        m.to_streamlit(height=500)

with col_report:
    st.subheader("Risk Report")
    if regions:
        selected = st.selectbox(
            "Select region",
            options=[r["name"] for r in regions],
            index=0,
        )
        region = next(r for r in regions if r["name"] == selected)

        tier = region.get("risk_tier")
        score = region.get("composite_score")
        if tier:
            icon = TIER_COLORS.get(tier, "⚪")
            st.metric(
                "Risk Tier",
                f"{icon} {tier.upper()}",
                delta=f"Score: {score:.2f}" if score else None,
            )

        if st.button("▶ Run Assessment", type="primary"):
            st.info(
                "Pipeline orchestration will be wired in Phase 3. "
                "Run `ingest_sentinel2_flow`, `ingest_noaa_flow`, and the LangGraph agents manually for now."
            )

        report = asyncio.run(get_report(region["id"]))
        if report:
            if report.get("low_confidence"):
                st.warning("⚠ Low confidence report — factuality score below threshold.")
            st.markdown(report["narrative"])
            if report.get("citations"):
                st.caption("**Sources:** " + " · ".join(
                    f"[{i+1}] {c}" for i, c in enumerate(report["citations"])
                ))
            if report.get("factuality_score") is not None:
                st.caption(f"Factuality score: {report['factuality_score']:.2f}")
        else:
            st.info("No report yet for this region. Click **Run Assessment** to generate one.")
