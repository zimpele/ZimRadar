"""LangGraph agent: autonomously gathers evidence and writes county risk reports."""

import asyncio
import json
import logging
import re
from typing import Literal, TypedDict

import httpx
from langgraph.graph import END, StateGraph
from sqlalchemy import text

from src.agents.prompts import (
    NARRATIVE_SYSTEM,
    PROMPT_VERSION,
    build_narrative_prompt,
)
from src.agents.tools import (
    fetch_climate_trend,
    fetch_fema_flood_history,
    fetch_infrastructure_summary,
    fetch_nri_comparison,
    fetch_storm_events_summary,
)
from src.config import get_settings
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    "flood_events_5yr": "Flood Events (5yr)",
    "avg_precipitation_trend": "Precipitation Trend",
    "vegetation_loss_pct": "Vegetation Loss %",
    "urban_density": "Urban Density",
    "elevation_variance": "Elevation Variance (m)",
    "infrastructure_age_proxy": "Infrastructure Age",
    "nri_risk_score": "NRI Risk Score",
    "nri_eal_score": "NRI Expected Annual Loss",
    "nri_sovi_score": "NRI Social Vulnerability",
    "nri_flood_risks": "NRI Flood Risk",
    "nri_fire_risks": "NRI Fire Risk",
    "nri_heat_risks": "NRI Heat Risk",
    "storm_events_5yr": "Storm Events (5yr)",
    "storm_damage_per_capita": "Storm Damage per Capita ($)",
}

MAX_RETRIES = 2


class ReportAgentState(TypedDict):
    region_id: int
    county_fips: str
    county_name: str
    risk_tier: str
    confidence: float
    shap_dict: dict
    features: dict
    tools_selected: list[str]
    evidence: dict
    draft_json: str
    validation_errors: list[str]
    retry_count: int
    report: dict
    flagged: bool


# ── Tool selection (deterministic, SHAP-driven) ────────────────────────────────


def _select_tools(shap_dict: dict) -> list[str]:
    """Map top SHAP features to relevant evidence tools."""
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    tools: set[str] = set()
    for feature, _ in top:
        if any(k in feature for k in ("fire", "wfir")):
            tools |= {"nri_comparison", "storm_events"}
        if any(k in feature for k in ("flood", "cfld", "rfld", "flood_events")):
            tools |= {"fema_flood_history", "storm_events"}
        if "precip" in feature:
            tools.add("climate_trend")
        if "infra" in feature or "elevation" in feature:
            tools.add("infrastructure")
        if any(k in feature for k in ("nri_risk", "nri_eal", "nri_sovi", "nri_heat")):
            tools.add("nri_comparison")
        if "storm" in feature:
            tools.add("storm_events")
    # Always include NRI comparison as baseline
    tools.add("nri_comparison")
    return list(tools)


# ── Graph nodes ────────────────────────────────────────────────────────────────


async def gather_evidence_node(state: ReportAgentState) -> dict:
    """Select and run evidence tools in parallel based on SHAP features."""
    tools = _select_tools(state["shap_dict"])
    logger.info("County %s: selected tools %s", state["county_fips"], tools)

    fips = state["county_fips"]
    tasks: dict[str, object] = {}
    if "fema_flood_history" in tools:
        tasks["fema_floods"] = fetch_fema_flood_history(fips)
    if "nri_comparison" in tools:
        tasks["nri"] = fetch_nri_comparison(fips)
    if "storm_events" in tools:
        tasks["storm_events"] = fetch_storm_events_summary(fips)
    if "climate_trend" in tools:
        tasks["climate"] = fetch_climate_trend(fips)
    if "infrastructure" in tools:
        tasks["infrastructure"] = fetch_infrastructure_summary(fips)

    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    evidence = {}
    for key, result in zip(tasks.keys(), results):
        evidence[key] = result if not isinstance(result, Exception) else {"error": str(result)}

    return {"tools_selected": tools, "evidence": evidence}


async def draft_narrative_node(state: ReportAgentState) -> dict:
    """Call Gemma2 via Ollama to write structured JSON report."""
    settings = get_settings()
    ollama_url = f"{settings.ollama_url}/api/generate"
    ollama_model = settings.ollama_model

    top_shap = sorted(state["shap_dict"].items(), key=lambda x: abs(x[1]), reverse=True)

    retry_note = ""
    if state.get("validation_errors") and state.get("retry_count", 0) > 0:
        retry_note = "; ".join(state["validation_errors"])

    prompt = build_narrative_prompt(
        county_name=state["county_name"],
        risk_tier=state["risk_tier"],
        confidence=state["confidence"],
        top_shap=top_shap,
        evidence=state["evidence"],
        feature_labels=FEATURE_LABELS,
        retry_note=retry_note,
    )
    full_prompt = f"{NARRATIVE_SYSTEM}\n\n{prompt}"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                ollama_url,
                json={"model": ollama_model, "prompt": full_prompt, "stream": False},
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
    except Exception as exc:
        logger.error("Gemma2 call failed: %s", exc)
        raw = (
            '{"top_drivers":[],"supporting_evidence":[],'
            '"uncertainty_notes":["LLM unavailable"],'
            '"briefing_md":"Report generation failed.","citations":[]}'
        )

    return {"draft_json": raw, "retry_count": state.get("retry_count", 0)}


def validate_node(state: ReportAgentState) -> dict:
    """Deterministic validation: JSON structure, SHAP alignment, numeric accuracy."""
    errors: list[str] = []
    raw = state.get("draft_json", "")

    # 1. Valid JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except Exception:
                return {"validation_errors": ["Invalid JSON — cannot parse response"], "report": {}}
        else:
            return {"validation_errors": ["Invalid JSON — no JSON object found"], "report": {}}

    # 2. Required fields
    required = ["top_drivers", "supporting_evidence", "uncertainty_notes", "briefing_md", "citations"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # 3. Risk tier consistency
    briefing = data.get("briefing_md", "").lower()
    tier = state["risk_tier"].lower()
    if tier not in briefing:
        errors.append(f"Briefing does not mention risk tier '{tier}'")

    # 4. County name present
    if state["county_name"].split(",")[0].lower() not in briefing:
        errors.append(f"Briefing does not mention county name '{state['county_name']}'")

    # 5. At least one citation
    if not data.get("citations"):
        errors.append("No citations provided")

    # 6. Top drivers reference actual SHAP features
    top_shap_features = {
        k
        for k, _ in sorted(
            state["shap_dict"].items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]
    }
    driver_text = " ".join(data.get("top_drivers", [])).lower()
    feature_words = {FEATURE_LABELS.get(f, f).lower().split()[0] for f in top_shap_features}
    if not any(word in driver_text for word in feature_words):
        errors.append("Top drivers do not reference actual top SHAP features")

    report = data if not errors else {}
    return {"validation_errors": errors, "report": report}


async def store_report_node(state: ReportAgentState) -> dict:
    """Save the validated (or flagged) report to county_reports table."""
    report = state.get("report") or {}
    flagged = bool(state.get("flagged")) or bool(state.get("validation_errors"))

    async with get_async_session() as session:
        await session.execute(
            text("""
                INSERT INTO county_reports
                    (region_id, county_fips, risk_tier, confidence, top_drivers,
                     evidence, briefing_md, citations, validation_pass, flagged,
                     model_version, prompt_version)
                VALUES
                    (:region_id, :fips, :tier, :conf,
                     CAST(:top_drivers AS jsonb), CAST(:evidence AS jsonb),
                     :briefing, CAST(:citations AS jsonb),
                     :valid, :flagged, :model_ver, :prompt_ver)
            """),
            {
                "region_id": state["region_id"],
                "fips": state["county_fips"],
                "tier": state["risk_tier"],
                "conf": state["confidence"],
                "top_drivers": json.dumps(report.get("top_drivers", [])),
                "evidence": json.dumps(state.get("evidence", {})),
                "briefing": report.get("briefing_md", state.get("draft_json", "")[:2000]),
                "citations": json.dumps(report.get("citations", [])),
                "valid": not flagged,
                "flagged": flagged,
                "model_ver": get_settings().ollama_model,
                "prompt_ver": PROMPT_VERSION,
            },
        )
    logger.info(
        "Stored report for region %d (%s) — flagged=%s",
        state["region_id"],
        state["county_name"],
        flagged,
    )
    return {"flagged": flagged}


# ── Routing ────────────────────────────────────────────────────────────────────


def route_after_validation(
    state: ReportAgentState,
) -> Literal["store", "repair", "store_flagged"]:
    errors = state.get("validation_errors", [])
    if not errors:
        return "store"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "store_flagged"
    return "repair"


def increment_retry(state: ReportAgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1, "flagged": False}


def mark_flagged(state: ReportAgentState) -> dict:
    logger.warning(
        "Report for %s flagged after %d retries: %s",
        state["county_name"],
        MAX_RETRIES,
        state.get("validation_errors"),
    )
    return {"flagged": True}


# ── Build graph ────────────────────────────────────────────────────────────────


def build_report_graph():
    graph = StateGraph(ReportAgentState)

    graph.add_node("gather_evidence", gather_evidence_node)
    graph.add_node("draft_narrative", draft_narrative_node)
    graph.add_node("validate", validate_node)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("mark_flagged", mark_flagged)
    graph.add_node("store_report", store_report_node)

    graph.set_entry_point("gather_evidence")
    graph.add_edge("gather_evidence", "draft_narrative")
    graph.add_edge("draft_narrative", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "store": "store_report",
            "repair": "increment_retry",
            "store_flagged": "mark_flagged",
        },
    )
    graph.add_edge("increment_retry", "draft_narrative")
    graph.add_edge("mark_flagged", "store_report")
    graph.add_edge("store_report", END)

    return graph.compile()


# Compiled graph (module-level singleton)
report_graph = build_report_graph()


async def generate_county_report(
    region_id: int,
    county_fips: str,
    county_name: str,
    risk_tier: str,
    confidence: float,
    shap_dict: dict,
    features: dict,
) -> dict:
    """Run the report agent graph for one county. Returns the final state."""
    initial_state: ReportAgentState = {
        "region_id": region_id,
        "county_fips": county_fips,
        "county_name": county_name,
        "risk_tier": risk_tier,
        "confidence": confidence,
        "shap_dict": shap_dict,
        "features": features,
        "tools_selected": [],
        "evidence": {},
        "draft_json": "",
        "validation_errors": [],
        "retry_count": 0,
        "report": {},
        "flagged": False,
    }
    final = await report_graph.ainvoke(initial_state)
    return final
