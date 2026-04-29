from typing import Any
from typing_extensions import TypedDict


class ZimRadarState(TypedDict, total=False):
    region_query: str
    region_id: int
    tile_paths: list[str]
    segmentation_results: dict[str, Any]
    depth_map: dict[str, Any]
    forecast: dict[str, Any]
    risk_tier: str
    risk_score: float
    retrieved_context: list[dict[str, Any]]
    report_draft: str
    citations: list[dict[str, Any]]
    factuality_score: float
    retry_count: int
    final_report: str | None
    report_id: str | None
    low_confidence: bool
