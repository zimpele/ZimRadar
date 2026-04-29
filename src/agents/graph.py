from langgraph.graph import StateGraph, END
from src.agents.state import ZimRadarState
from src.agents.ingest import ingest_node
from src.agents.analysis import analysis_node
from src.agents.report import report_node
from src.agents.validator import validator_node


def _route_after_validator(state: ZimRadarState) -> str:
    if state.get("final_report") is not None:
        return END
    return "report"


def build_graph():
    g = StateGraph(ZimRadarState)

    g.add_node("ingest", ingest_node)
    g.add_node("analysis", analysis_node)
    g.add_node("report", report_node)
    g.add_node("validator", validator_node)

    g.set_entry_point("ingest")
    g.add_edge("ingest", "analysis")
    g.add_edge("analysis", "report")
    g.add_edge("report", "validator")
    g.add_conditional_edges("validator", _route_after_validator)

    return g.compile()
