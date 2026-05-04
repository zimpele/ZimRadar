"""ZimRadar Streamlit dashboard — interactive climate risk visualization."""

import asyncio
import json
import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from src.config import get_settings
from src.pipeline.classifier import FEATURE_NAMES, RISK_TIERS, load_classifier_from_s3
from src.storage.db import get_async_session

# ── Page config ───────────────────────────────────────────────────────────────

_settings = get_settings()
if _settings.langsmith_api_key:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", _settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", _settings.langsmith_project)

st.set_page_config(page_title="ZimRadar", layout="wide", page_icon="🌍")
st.title("🌍 ZimRadar — Climate Risk Intelligence")

US_STATES_GEOJSON = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)

TIER_EMOJI = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}
TIER_COLOR_MAP = {
    "critical": "#d73027",
    "high": "#fc8d59",
    "moderate": "#fee08b",
    "low": "#91cf60",
}
FEATURE_LABELS = {
    "flood_events_5yr": "Flood Events (5yr)",
    "avg_precipitation_trend": "Precipitation Trend",
    "vegetation_loss_pct": "Vegetation Loss %",
    "urban_density": "Urban Density",
    "elevation_variance": "Elevation Variance (m)",
    "infrastructure_age_proxy": "Infrastructure Age",
    "nri_risk_score": "NRI Risk Score",
    "nri_eal_score": "NRI EAL Score",
    "nri_sovi_score": "NRI Social Vulnerability",
    "nri_flood_risks": "NRI Flood Risk",
    "nri_fire_risks": "NRI Fire Risk",
    "nri_heat_risks": "NRI Heat Risk",
}

# ── Cached resources ──────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def _load_model():
    return load_classifier_from_s3()


@st.cache_data(ttl=86400, show_spinner=False)
def _load_us_states() -> dict | None:
    import httpx as _httpx

    try:
        resp = _httpx.get(US_STATES_GEOJSON, timeout=15.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_county_names() -> dict[str, list[tuple[str, str]]]:
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


# ── DB helpers ────────────────────────────────────────────────────────────────


async def get_regions() -> list[dict]:
    async with get_async_session() as session:
        result = await session.execute(
            text("""
                SELECT r.id, r.name, r.bbox, r.geometry, r.state_code, r.county_fips,
                       ra.risk_tier, ra.composite_score, ra.confidence,
                       ra.assessed_at, ra.features_json, ra.shap_values
                FROM regions r
                LEFT JOIN LATERAL (
                    SELECT risk_tier, composite_score, confidence, assessed_at,
                           features_json, shap_values
                    FROM risk_assessments
                    WHERE region_id = r.id
                    ORDER BY assessed_at DESC LIMIT 1
                ) ra ON TRUE
                WHERE r.active = TRUE
                ORDER BY r.name
            """)
        )
        return [dict(row._mapping) for row in result.fetchall()]


async def get_risk_stats() -> dict:
    async with get_async_session() as session:
        total = (
            await session.execute(text("SELECT COUNT(*) FROM regions WHERE active = TRUE"))
        ).scalar()
        rows = await session.execute(
            text("""
                SELECT risk_tier, COUNT(*) FROM (
                    SELECT DISTINCT ON (region_id) risk_tier
                    FROM risk_assessments
                    ORDER BY region_id, assessed_at DESC
                ) t GROUP BY risk_tier
            """)
        )
        counts = {row[0]: int(row[1]) for row in rows.fetchall()}
    return {"total": int(total or 0), **counts}


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


# ── Chart helpers ─────────────────────────────────────────────────────────────


def _shap_chart(shap_dict: dict) -> go.Figure:
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]))
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_layout(
        title="SHAP — Feature Contributions to Risk Prediction",
        xaxis_title="Impact on risk score (red = increases risk)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _importance_chart() -> go.Figure | None:
    try:
        model = _load_model()
        importances = model.feature_importances_
        idx = np.argsort(importances)
        labels = [FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in idx]
        vals = importances[idx]
        fig = go.Figure(
            go.Bar(
                x=vals,
                y=labels,
                orientation="h",
                marker_color="#2196F3",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Global Feature Importance (XGBoost gain)",
            xaxis_title="Importance",
            height=440,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    except Exception:
        return None


def _distribution_chart(stats: dict) -> go.Figure:
    tiers = [t for t in RISK_TIERS if t in stats]
    vals = [stats[t] for t in tiers]
    colors = [TIER_COLOR_MAP[t] for t in tiers]
    fig = go.Figure(
        go.Pie(
            labels=[t.title() for t in tiers],
            values=vals,
            marker_colors=colors,
            hole=0.42,
            hovertemplate="%{label}: %{value} counties<extra></extra>",
        )
    )
    fig.update_layout(
        title="Risk Tier Distribution",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

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

# ── Top metrics ───────────────────────────────────────────────────────────────

stats = asyncio.run(get_risk_stats())
m_cols = st.columns(5)
m_cols[0].metric("Tracked Counties", stats["total"])
for i, tier in enumerate(RISK_TIERS):
    m_cols[i + 1].metric(
        f"{TIER_EMOJI.get(tier, '')} {tier.title()}",
        stats.get(tier, 0),
    )

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_map, tab_insights = st.tabs(["🗺️ Map & County Analysis", "📊 Model Insights"])

regions = asyncio.run(get_regions())

# ── Tab 1: Map & County Analysis ──────────────────────────────────────────────

with tab_map:
    col_map, col_detail = st.columns([3, 2])

    with col_map:
        st.subheader("Risk Map")
        if not regions:
            st.info("No regions tracked yet. Use **Add County** in the sidebar to get started.")
        else:
            import folium
            import leafmap.foliumap as leafmap

            m = leafmap.Map(center=[39, -98], zoom=4)

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

            us_bounds = []
            for region in regions:
                tier = region.get("risk_tier", "unknown")
                color = TIER_COLOR_MAP.get(tier, "#aaaaaa")
                score = region.get("composite_score")
                conf = region.get("confidence")
                label = region["name"]
                if tier and tier != "unknown":
                    label += f" — {tier.upper()}"
                if score is not None:
                    label += f" (score: {score:.2f}"
                    if conf is not None:
                        label += f", {conf:.0%} conf"
                    label += ")"

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

                if region.get("county_fips"):
                    raw_bbox = region.get("bbox") or {}
                    bbox = json.loads(raw_bbox) if isinstance(raw_bbox, str) else raw_bbox
                    if all(k in bbox for k in ("min_lon", "min_lat", "max_lon", "max_lat")):
                        us_bounds.append([bbox["min_lat"], bbox["min_lon"]])
                        us_bounds.append([bbox["max_lat"], bbox["max_lon"]])

            if us_bounds:
                m.fit_bounds(us_bounds)
            m.to_streamlit(height=500)

    with col_detail:
        st.subheader("County Analysis")
        if not regions:
            st.info("Add a county to see analysis.")
        else:
            selected = st.selectbox(
                "Select county",
                options=[r["name"] for r in regions],
                index=0,
            )
            region = next(r for r in regions if r["name"] == selected)

            tier = region.get("risk_tier")
            score = region.get("composite_score")
            conf = region.get("confidence")

            if tier:
                icon = TIER_EMOJI.get(tier, "⚪")
                c1, c2 = st.columns(2)
                c1.metric("Risk Tier", f"{icon} {tier.upper()}")
                c2.metric("Composite Score", f"{score:.3f}" if score is not None else "—")
                if conf is not None:
                    st.progress(float(conf), text=f"Model confidence: {conf:.0%}")
            else:
                st.info("Not yet assessed — click **Run Assessment** below.")

            # Feature values
            raw_features = region.get("features_json")
            features = json.loads(raw_features) if isinstance(raw_features, str) else raw_features
            if features:
                with st.expander("📋 Feature Values", expanded=False):
                    for k in FEATURE_NAMES:
                        v = features.get(k)
                        label = FEATURE_LABELS.get(k, k)
                        display = f"`{v:.4f}`" if isinstance(v, float) else f"`{v}`"
                        st.markdown(f"**{label}:** {display}")

            # SHAP waterfall
            raw_shap = region.get("shap_values")
            shap_dict = json.loads(raw_shap) if isinstance(raw_shap, str) else raw_shap
            if shap_dict:
                st.plotly_chart(_shap_chart(shap_dict), use_container_width=True)
            elif tier:
                st.caption("SHAP explanations will appear after the next assessment run.")

            st.divider()

            # Run Assessment
            if st.button(
                "🔮 Run Assessment",
                type="primary",
                help=f"Local model: {_settings.ollama_model}",
            ):
                import asyncio as _asyncio
                from src.agents.graph import build_graph

                with st.status("Running assessment…", expanded=True) as status:
                    st.write("Resolving region and ingesting data…")
                    graph = build_graph()
                    try:
                        final_state = _asyncio.run(
                            graph.ainvoke(
                                {
                                    "region_query": selected,
                                    "retry_count": 0,
                                    "final_report": None,
                                    "low_confidence": False,
                                }
                            )
                        )
                        status.update(label="Assessment complete!", state="complete")
                    except Exception as exc:
                        status.update(label="Assessment failed", state="error")
                        st.error(str(exc))
                        st.stop()

                if final_state.get("final_report"):
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Risk Tier", final_state.get("risk_tier", "—").upper())
                    r2.metric("Score", f"{final_state.get('risk_score', 0):.2f}")
                    r3.metric(
                        "Factuality",
                        f"{final_state.get('factuality_score', 0):.2f}",
                        delta="⚠ low confidence" if final_state.get("low_confidence") else None,
                    )
                    st.markdown(final_state["final_report"])
                    st.rerun()

            # Saved narrative report
            report = asyncio.run(get_report(region["id"]))
            if report:
                if report.get("low_confidence"):
                    st.warning("⚠ Low confidence — factuality score below threshold.")
                st.markdown(report["narrative"])
                if report.get("citations"):
                    st.caption(
                        "**Sources:** "
                        + " · ".join(f"[{i + 1}] {c}" for i, c in enumerate(report["citations"]))
                    )
                if report.get("factuality_score") is not None:
                    st.caption(f"Factuality score: {report['factuality_score']:.2f}")
            elif not tier:
                st.info("No report yet. Click **Run Assessment** to generate one.")

# ── Tab 2: Model Insights ─────────────────────────────────────────────────────

with tab_insights:
    st.subheader("Model & Data Insights")
    ins_left, ins_right = st.columns([3, 2])

    with ins_left:
        fig_imp = _importance_chart()
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance unavailable — train the classifier first.")

    with ins_right:
        if any(t in stats for t in RISK_TIERS):
            st.plotly_chart(_distribution_chart(stats), use_container_width=True)
        else:
            st.info("No assessed counties yet.")

        assessed = [r for r in regions if r.get("risk_tier")]
        if assessed:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "County": r["name"],
                        "Tier": r["risk_tier"].upper(),
                        "Score": round(r["composite_score"], 3)
                        if r.get("composite_score")
                        else None,
                        "Conf.": f"{r['confidence']:.0%}" if r.get("confidence") else "—",
                    }
                    for r in sorted(
                        assessed,
                        key=lambda x: x.get("composite_score") or 0,
                        reverse=True,
                    )
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
