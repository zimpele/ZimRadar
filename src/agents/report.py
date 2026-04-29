from src.agents.llm import complete
from src.agents.state import ZimRadarState
from src.rag.retriever import retrieve


async def report_node(state: ZimRadarState) -> ZimRadarState:
    region_query = state.get("region_query", "")
    risk_tier = state.get("risk_tier", "moderate")
    risk_score = state.get("risk_score", 0.5)
    forecast = state.get("forecast", {})
    retry_count = state.get("retry_count", 0)

    context_docs = await retrieve(f"climate risk {region_query} flood fire disaster")

    citations = [
        {
            "index": i + 1,
            "text": doc["text"][:200],
            "source_type": doc.get("source_type", "unknown"),
            "source_id": doc.get("source_id", ""),
        }
        for i, doc in enumerate(context_docs)
    ]

    context_text = "\n\n".join(f"[{i + 1}] {doc['text']}" for i, doc in enumerate(context_docs))

    prompt = f"""Generate a climate risk assessment report for: {region_query}

Risk Assessment:
- Risk Tier: {risk_tier}
- Composite Score: {risk_score:.2f}
- Flood Risk Flag: {forecast.get("flood_risk_flag", False)}
- Fire Risk Flag: {forecast.get("fire_risk_flag", False)}

Retrieved Context:
{context_text if context_text else "No historical records found."}

Write a concise 3-paragraph narrative with inline citations in [n] format.
Paragraph 1: Current risk factors. Paragraph 2: Historical trends and forecasts.
Paragraph 3: Recommended actions for insurers and municipal planners."""

    system = (
        "You are a climate risk analyst. Write factual, citation-grounded reports. "
        "Only cite sources that appear in the Retrieved Context above."
    )

    narrative = await complete(prompt, system=system)

    return {
        **state,
        "retrieved_context": context_docs,
        "report_draft": narrative,
        "citations": citations,
        "retry_count": retry_count,
    }
