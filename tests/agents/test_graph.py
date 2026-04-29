from src.agents.graph import build_graph
from src.agents.state import ZimRadarState


def test_graph_compiles_without_error():
    graph = build_graph()
    assert graph is not None


def test_graph_has_required_nodes():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert "ingest" in node_names
    assert "analysis" in node_names
    assert "report" in node_names
    assert "validator" in node_names


def test_initial_state_shape():
    state: ZimRadarState = {
        "region_query": "Dallas County, TX",
        "retry_count": 0,
        "final_report": None,
        "low_confidence": False,
    }
    assert state["region_query"] == "Dallas County, TX"
