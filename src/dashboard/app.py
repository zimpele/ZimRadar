import asyncio
import json
import os
import streamlit as st
from sqlalchemy import text
from src.config import get_settings
from src.storage.db import get_async_session

# Enable LangSmith tracing if key is configured
_settings = get_settings()
if _settings.langsmith_api_key:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", _settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", _settings.langsmith_project)

st.set_page_config(page_title="ZimRadar", layout="wide", page_icon="🌍")
st.title("🌍 ZimRadar — Climate Risk Assessment")

US_STATES_GEOJSON = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)


async def get_regions() -> list[dict]:
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT r.id, r.name, r.bbox, r.geometry, r.state_code, r.county_fips,
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


@st.cache_data(ttl=86400)
def _load_us_states() -> dict | None:
    """Fetch US state boundary GeoJSON (cached for 24 h)."""
    import httpx as _httpx

    try:
        resp = _httpx.get(US_STATES_GEOJSON, timeout=15.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _load_county_names() -> dict[str, list[tuple[str, str]]]:
    """Return {state_code: [(county_name, fips), ...]} for all US counties."""
    from src.ingestion.geo_admin import FIPS_TO_STATE, _get_county_features

    features = asyncio.run(_get_county_features())
    by_state: dict[str, list[tuple[str, str]]] = {}
    for fips, entry in features.items():
        sc = FIPS_TO_STATE.get(fips[:2])
        if sc:
            by_state.setdefault(sc, []).append((entry["name"], fips))
    for sc in by_state:
        by_state[sc].sort()
    return by_state


TIER_COLORS = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}
TIER_COLOR_MAP = {
    "critical": "#d73027",
    "high": "#fc8d59",
    "moderate": "#fee08b",
    "low": "#91cf60",
}

# ── Sidebar: Add County ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("Add County")
    with st.expander("➕ Add new county"):
        with st.spinner("Loading county list…"):
            county_names = _load_county_names()
        state_list = sorted(county_names.keys())
        sel_state = st.selectbox("State", state_list, key="new_county_state")
        counties_in_state = county_names.get(sel_state, [])
        county_labels = [name for name, _ in counties_in_state]
        sel_idx = st.selectbox(
            "County",
            range(len(county_labels)),
            format_func=lambda i: county_labels[i],
            key="new_county_name",
        )
        if st.button("Add County", key="add_county_btn"):
            if counties_in_state:
                _, fips = counties_in_state[sel_idx]
                from src.ingestion.geo_admin import add_county_region

                try:
                    region_id = asyncio.run(add_county_region(fips))
                    st.success(f"Added region #{region_id}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

# ── Main layout ──────────────────────────────────────────────────────────────

col_map, col_report = st.columns([3, 2])

with col_map:
    st.subheader("Tracked Regions")
    regions = asyncio.run(get_regions())

    if not regions:
        st.info("No regions tracked yet. Use **Add County** in the sidebar to get started.")
    else:
        import folium
        import leafmap.foliumap as leafmap

        # US-centered default view; county fit_bounds will refine if needed.
        m = leafmap.Map(center=[39, -98], zoom=4)

        # Layer 1: US state outlines (thin gray, no fill) for geographic context.
        states_geojson = _load_us_states()
        if states_geojson:
            folium.GeoJson(
                states_geojson,
                name="US States",
                style_function=lambda _: {
                    "fillColor": "transparent",
                    "color": "#888888",
                    "weight": 0.8,
                    "fillOpacity": 0,
                },
                tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["State:"]),
            ).add_to(m)

        # Layer 2: Assessed county polygons colored by risk tier.
        us_bounds = []
        for region in regions:
            tier = region.get("risk_tier", "unknown")
            color = TIER_COLOR_MAP.get(tier, "#aaaaaa")
            score = region.get("composite_score")
            label = region["name"]
            if tier and tier != "unknown":
                label += f" — {tier.upper()}"
            if score is not None:
                label += f" ({score:.2f})"

            raw_geom = region.get("geometry")
            geom = json.loads(raw_geom) if isinstance(raw_geom, str) else raw_geom

            if geom:
                folium.GeoJson(
                    {"type": "Feature", "geometry": geom, "properties": {}},
                    style_function=lambda _, c=color: {
                        "fillColor": c,
                        "color": c,
                        "weight": 1.5,
                        "fillOpacity": 0.45,
                    },
                    tooltip=label,
                ).add_to(m)
            else:
                # Fallback rectangle for regions without geometry yet.
                raw_bbox = region.get("bbox") or {}
                bbox = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
                if all(k in bbox for k in ("min_lon", "min_lat", "max_lon", "max_lat")):
                    folium.Rectangle(
                        bounds=[
                            [bbox["min_lat"], bbox["min_lon"]],
                            [bbox["max_lat"], bbox["max_lon"]],
                        ],
                        color=color,
                        fill=True,
                        fill_opacity=0.45,
                        weight=1.5,
                        tooltip=label,
                    ).add_to(m)

            # Collect US county bounds for auto-zoom (skip non-US regions).
            if region.get("county_fips"):
                raw_bbox = region.get("bbox") or {}
                bbox = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
                if all(k in bbox for k in ("min_lon", "min_lat", "max_lon", "max_lat")):
                    us_bounds.append([bbox["min_lat"], bbox["min_lon"]])
                    us_bounds.append([bbox["max_lat"], bbox["max_lon"]])

        if us_bounds:
            m.fit_bounds(us_bounds)

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

        region_name = selected
        if st.button("Run Assessment", type="primary"):
            if not region_name:
                st.error("Enter a region name first.")
            else:
                import asyncio as _asyncio
                from src.agents.graph import build_graph

                with st.status("Running assessment…", expanded=True) as status:
                    st.write("Resolving region and ingesting data…")
                    initial_state = {
                        "region_query": region_name,
                        "retry_count": 0,
                        "final_report": None,
                        "low_confidence": False,
                    }
                    graph = build_graph()
                    try:
                        final_state = _asyncio.run(graph.ainvoke(initial_state))
                        status.update(label="Assessment complete!", state="complete")
                    except Exception as exc:
                        status.update(label="Assessment failed", state="error")
                        st.error(str(exc))
                        st.stop()

                if final_state.get("final_report"):
                    st.subheader("Risk Assessment")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Tier", final_state.get("risk_tier", "—").upper())
                    col2.metric("Score", f"{final_state.get('risk_score', 0):.2f}")
                    col3.metric(
                        "Factuality",
                        f"{final_state.get('factuality_score', 0):.2f}",
                        delta="⚠ low confidence" if final_state.get("low_confidence") else None,
                    )
                    st.subheader("Narrative Report")
                    st.markdown(final_state["final_report"])
                    if final_state.get("citations"):
                        st.subheader("Sources")
                        for c in final_state["citations"]:
                            st.caption(
                                f"[{c['index']}] {c.get('source_type', 'unknown')} — "
                                f"{c.get('source_id', '')}: {c.get('text', '')}"
                            )

        report = asyncio.run(get_report(region["id"]))
        if report:
            if report.get("low_confidence"):
                st.warning("⚠ Low confidence report — factuality score below threshold.")
            st.markdown(report["narrative"])
            if report.get("citations"):
                st.caption(
                    "**Sources:** "
                    + " · ".join(f"[{i + 1}] {c}" for i, c in enumerate(report["citations"]))
                )
            if report.get("factuality_score") is not None:
                st.caption(f"Factuality score: {report['factuality_score']:.2f}")
        else:
            st.info("No report yet for this region. Click **Run Assessment** to generate one.")
