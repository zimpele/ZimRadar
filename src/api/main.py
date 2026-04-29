import uuid
import asyncio
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from src.agents.graph import build_graph
from src.config import get_settings
from src.storage.db import get_async_session
from src.storage.models import Report

app = FastAPI(title="ZimRadar API", version="0.1.0")

_graph = None
_graph_lock = asyncio.Lock()


async def _get_graph():
    global _graph
    if _graph is None:
        async with _graph_lock:
            if _graph is None:
                _graph = build_graph()
    return _graph


class AssessRequest(BaseModel):
    region: str
    date_range: list[str] | None = None


@app.post("/assess")
async def assess(req: AssessRequest, authorization: str | None = Header(None)):
    settings = get_settings()
    if settings.api_key and authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    graph = await _get_graph()
    initial_state = {
        "region_query": req.region,
        "retry_count": 0,
        "final_report": None,
        "low_confidence": False,
    }
    final_state = await graph.ainvoke(initial_state)

    report_id = final_state.get("report_id")
    if not report_id:
        raise HTTPException(status_code=500, detail="Report generation failed")

    return {"report_id": report_id}


@app.get("/report/{report_id}")
async def get_report(report_id: str):
    try:
        rid = uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report ID format")

    async with get_async_session() as session:
        report = await session.get(Report, rid)

    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "report_id": str(report.id),
        "region_id": report.region_id,
        "narrative": report.narrative,
        "citations": report.citations,
        "factuality_score": report.factuality_score,
        "low_confidence": report.low_confidence,
        "retry_count": report.retry_count,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
