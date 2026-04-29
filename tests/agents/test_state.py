from src.agents.state import ZimRadarState


def test_state_is_a_typed_dict():
    state: ZimRadarState = {}
    assert isinstance(state, dict)


def test_state_can_hold_all_fields():
    state: ZimRadarState = {
        "region_query": "Harris County, TX",
        "region_id": 1,
        "tile_paths": ["s3://bucket/tile1.tif"],
        "segmentation_results": {},
        "depth_map": {},
        "forecast": {"flood_risk_flag": True},
        "risk_tier": "high",
        "risk_score": 0.75,
        "retrieved_context": [],
        "report_draft": "draft text",
        "citations": [],
        "factuality_score": 0.9,
        "retry_count": 0,
        "final_report": "final text",
        "report_id": None,
        "low_confidence": False,
    }
    assert state["region_query"] == "Harris County, TX"
    assert state["risk_tier"] == "high"
    assert state["low_confidence"] is False
