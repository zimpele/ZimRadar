import re
import uuid
from datetime import datetime, timezone
from src.agents.llm import complete
from src.agents.state import ZimRadarState
from src.storage.db import get_async_session
from src.storage.models import Report

MAX_RETRIES = 2
FACTUALITY_THRESHOLD = 0.8


async def validator_node(state: ZimRadarState) -> ZimRadarState:
    report_draft = state.get("report_draft", "")
    context_docs = state.get("retrieved_context", [])
    retry_count = state.get("retry_count", 0)
    region_id = state.get("region_id", 0)
    citations = state.get("citations", [])

    context_snippets = "\n".join(
        f"[{i + 1}] {doc['text'][:300]}" for i, doc in enumerate(context_docs[:5])
    )

    prompt = f"""Score the factuality of this climate risk report from 0.0 to 1.0.
1.0 = every factual claim is grounded in the provided context.
0.0 = the report contains fabricated facts not in the context.

Context:
{context_snippets if context_snippets else "(none)"}

Report:
{report_draft}

Respond with ONLY a decimal number between 0.0 and 1.0."""

    score_str = await complete(prompt)
    match = re.search(r"\d+\.\d+|\d+", score_str)
    factuality_score = float(match.group()) if match else 0.5
    factuality_score = max(0.0, min(1.0, factuality_score))

    should_finalize = factuality_score >= FACTUALITY_THRESHOLD or retry_count >= MAX_RETRIES

    if should_finalize:
        report_id = str(uuid.uuid4())
        report = Report(
            id=uuid.UUID(report_id),
            region_id=region_id,
            narrative=report_draft,
            citations=citations,
            factuality_score=factuality_score,
            retry_count=retry_count,
            low_confidence=factuality_score < FACTUALITY_THRESHOLD,
            created_at=datetime.now(timezone.utc),
        )
        async with get_async_session() as session:
            session.add(report)

        return {
            **state,
            "factuality_score": factuality_score,
            "final_report": report_draft,
            "report_id": report_id,
            "low_confidence": factuality_score < FACTUALITY_THRESHOLD,
        }

    return {
        **state,
        "factuality_score": factuality_score,
        "final_report": None,
        "retry_count": retry_count + 1,
    }
