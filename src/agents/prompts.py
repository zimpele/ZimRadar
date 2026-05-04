"""Prompts for the county report agent."""

PROMPT_VERSION = "v1"

NARRATIVE_SYSTEM = """You are a climate risk analyst writing county-level risk briefings.
Write in clear, factual language. Use only the evidence provided. Do not speculate.
Return a JSON object with exactly these fields:
{
  "top_drivers": ["<feature>: <plain-english explanation>", ...],  // 3 items
  "supporting_evidence": ["<factual claim from evidence>", ...],   // 3-5 items
  "uncertainty_notes": ["<limitation or caveat>", ...],            // 1-2 items
  "briefing_md": "<3 paragraph markdown briefing>",
  "citations": ["<source name>: <specific data point used>", ...]  // one per evidence source
}
Return only valid JSON. No prose outside the JSON object."""


def build_narrative_prompt(
    county_name: str,
    risk_tier: str,
    confidence: float,
    top_shap: list[tuple[str, float]],
    evidence: dict,
    feature_labels: dict[str, str],
    retry_note: str = "",
) -> str:
    shap_lines = "\n".join(
        f"  - {feature_labels.get(f, f)}: SHAP={v:+.4f}"
        f" ({'increases' if v > 0 else 'decreases'} risk)"
        for f, v in top_shap[:5]
    )
    evidence_lines = "\n".join(
        f"[{src}]\n{_fmt_dict(data)}"
        for src, data in evidence.items()
        if isinstance(data, dict) and data.get("available", True)
    )
    retry = (
        f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {retry_note}\nFix the issues above."
        if retry_note
        else ""
    )
    return (
        f"County: {county_name}\n"
        f"Risk tier: {risk_tier.upper()} (model confidence: {confidence:.1%})\n"
        f"\nTop risk drivers (SHAP attribution):\n{shap_lines}"
        f"\n\nEvidence gathered:\n{evidence_lines}"
        f"\n\nWrite a risk briefing for {county_name} explaining the {risk_tier} risk"
        f" classification.{retry}"
    )


def _fmt_dict(d: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in d.items() if k != "source" and v is not None)
