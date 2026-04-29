# ZimRadar Phase 3 — Multi-Agent Orchestration & API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire LangGraph (4-agent graph: Ingest → Analysis → Report → Validator) + OpenRouter free-model LLM + FastAPI REST endpoints to produce end-to-end climate risk reports.

**Architecture:** A `StateGraph` with four sequential nodes plus a conditional retry edge (Validator → Report if factuality < 0.8, max 2 retries). OpenRouter free models (httpx) handle LLM calls; Ollama is the local fallback when `OPENROUTER_API_KEY` is blank. FastAPI wraps the same graph. The Streamlit dashboard calls the graph directly as a Python function (no FastAPI dependency).

**Tech Stack:** LangGraph ≥ 0.2, FastAPI ≥ 0.111, uvicorn ≥ 0.30, httpx (existing), OpenRouter API (OpenAI-compatible chat completions), existing `src.pipeline.*`, `src.rag.retriever`, `src.storage.*`

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `src/config.py` | Modify | Add `openrouter_api_key`, `openrouter_model` settings |
| `pyproject.toml` | Modify | Add langgraph, fastapi, uvicorn to dependencies |
| `src/agents/__init__.py` | Create | Package marker |
| `src/agents/llm.py` | Create | Async LLM client: OpenRouter → Ollama fallback |
| `src/agents/state.py` | Create | `ZimRadarState` TypedDict |
| `src/agents/ingest.py` | Create | Ingest agent node: resolve region, load tile paths |
| `src/agents/analysis.py` | Create | Analysis agent node: forecast + classify + score |
| `src/agents/report.py` | Create | Report agent node: RAG retrieval + LLM narrative |
| `src/agents/validator.py` | Create | Validator agent node: LLM-as-judge + DB save |
| `src/agents/graph.py` | Create | `build_graph()` — compile StateGraph with all nodes |
| `src/api/__init__.py` | Create | Package marker |
| `src/api/main.py` | Create | FastAPI: `POST /assess`, `GET /report/{id}`, `GET /regions` |
| `src/dashboard/app.py` | Modify | "Run Assessment" button calls `build_graph().ainvoke()` |
| `Dockerfile` | Modify | Add `api` build target |
| `docker-compose.yml` | Modify | Add `api` service on port 8000 |
| `tests/agents/__init__.py` | Create | Package marker |
| `tests/agents/test_llm.py` | Create | LLM client unit tests (mock httpx) |
| `tests/agents/test_state.py` | Create | State shape validation |
| `tests/agents/test_ingest.py` | Create | Ingest node unit tests (mock DB) |
| `tests/agents/test_analysis.py` | Create | Analysis node unit tests (mock pipeline calls) |
| `tests/agents/test_report.py` | Create | Report node unit tests (mock retriever + LLM) |
| `tests/agents/test_validator.py` | Create | Validator node unit tests (mock LLM + DB) |
| `tests/agents/test_graph.py` | Create | Graph compilation smoke test |
| `tests/api/__init__.py` | Create | Package marker |
| `tests/api/test_main.py` | Create | FastAPI endpoint tests (mock graph) |

---

### Task 1: Config fields and dependencies

**Files:**
- Modify: `src/config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_config.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from src.config import Settings


def test_openrouter_fields_have_defaults():
    s = Settings()
    assert s.openrouter_api_key == ""
    assert s.openrouter_model == "meta-llama/llama-3.3-70b-instruct:free"


def test_openrouter_fields_read_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-3-27b-it:free")
    s = Settings()
    assert s.openrouter_api_key == "sk-test-key"
    assert s.openrouter_model == "google/gemma-3-27b-it:free"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `FAILED — AttributeError: 'Settings' object has no attribute 'openrouter_api_key'`

- [ ] **Step 3: Add fields to config**

In `src/config.py`, add after the `ollama_model` field:

```python
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
```

Full updated `src/config.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://zimradar:password@localhost:5432/zimradar"
    redis_url: str = "redis://localhost:6379"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma2:9b"

    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"

    sentinelsat_user: str = ""
    sentinelsat_pass: str = ""
    noaa_api_key: str = ""

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    s3_bucket_tiles: str = "zimradar-tiles"
    s3_bucket_pdfs: str = "zimradar-pdfs"

    langsmith_api_key: str = ""
    langsmith_project: str = "zimradar"
    api_key: str = ""

    risk_score_weights: str = "0.6,0.2,0.2"

    @property
    def risk_weights(self) -> tuple[float, float, float]:
        w = [float(x) for x in self.risk_score_weights.split(",")]
        return w[0], w[1], w[2]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Add langgraph, fastapi, uvicorn to pyproject.toml dependencies**

In `pyproject.toml`, add to the `dependencies` list:

```toml
  "langgraph>=0.2",
  "fastapi>=0.111",
  "uvicorn>=0.30",
  "python-multipart>=0.0.9",
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add src/config.py pyproject.toml tests/test_config.py
git commit -m "feat: add OpenRouter config fields and Phase 3 dependencies"
```

---

### Task 2: OpenRouter LLM client

**Files:**
- Create: `src/agents/__init__.py`
- Create: `src/agents/llm.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_llm.py
import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.llm import complete


@pytest.mark.asyncio
async def test_complete_uses_openrouter_when_key_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test response"}}]
    }

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        # Reset lru_cache so monkeypatch env var is picked up
        from src.config import get_settings
        get_settings.cache_clear()

        result = await complete("hello", system="be helpful")
        assert result == "test response"

        call_kwargs = mock_client.post.call_args
        assert "openrouter.ai" in call_kwargs[0][0]


@pytest.mark.asyncio
async def test_complete_falls_back_to_ollama_when_no_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"response": "ollama reply"}

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        from src.config import get_settings
        get_settings.cache_clear()

        result = await complete("hello")
        assert result == "ollama reply"

        call_kwargs = mock_client.post.call_args
        assert "ollama" in call_kwargs[0][0] or "11434" in call_kwargs[0][0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_llm.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.llm'`

- [ ] **Step 3: Create package markers and LLM client**

Create `src/agents/__init__.py` (empty file).

Create `tests/agents/__init__.py` (empty file).

Create `src/agents/llm.py`:

```python
import httpx
from src.config import get_settings


async def complete(prompt: str, system: str = "") -> str:
    settings = get_settings()

    if settings.openrouter_api_key:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/zimpele/ZimRadar",
                },
                json={"model": settings.openrouter_model, "messages": messages},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    # Ollama fallback
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/generate",
            json={"model": settings.ollama_model, "prompt": full_prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_llm.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agents/__init__.py src/agents/llm.py tests/agents/__init__.py tests/agents/test_llm.py
git commit -m "feat: add OpenRouter LLM client with Ollama fallback"
```

---

### Task 3: LangGraph state TypedDict

**Files:**
- Create: `src/agents/state.py`
- Create: `tests/agents/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_state.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.state'`

- [ ] **Step 3: Create state module**

Create `src/agents/state.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_state.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agents/state.py tests/agents/test_state.py
git commit -m "feat: add ZimRadarState TypedDict for LangGraph"
```

---

### Task 4: Ingest Agent node

**Files:**
- Create: `src/agents/ingest.py`
- Create: `tests/agents/test_ingest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_ingest.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.ingest import ingest_node


@pytest.mark.asyncio
async def test_ingest_node_sets_region_id_and_tile_paths():
    mock_region_row = MagicMock()
    mock_region_row.__getitem__ = lambda self, k: 42

    mock_tile_row = MagicMock()
    mock_tile_row.__getitem__ = lambda self, k: "s3://bucket/tile.tif"

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=[
        MagicMock(fetchone=MagicMock(return_value=(42,))),
        MagicMock(fetchall=MagicMock(return_value=[(("s3://bucket/tile.tif",),)])),
    ])

    with patch("src.agents.ingest.get_async_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_query": "Harris County"}
        result = await ingest_node(state)

    assert result["region_id"] == 42


@pytest.mark.asyncio
async def test_ingest_node_region_not_found():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(
        return_value=MagicMock(fetchone=MagicMock(return_value=None))
    )

    with patch("src.agents.ingest.get_async_session") as mock_ctx:
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_query": "Nonexistent County"}
        result = await ingest_node(state)

    assert "error" in result
    assert "Nonexistent County" in result["error"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_ingest.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.ingest'`

- [ ] **Step 3: Create ingest node**

Create `src/agents/ingest.py`:

```python
from sqlalchemy import text
from src.agents.state import ZimRadarState
from src.storage.db import get_async_session


async def ingest_node(state: ZimRadarState) -> ZimRadarState:
    region_query = state.get("region_query", "")

    async with get_async_session() as session:
        result = await session.execute(
            text("SELECT id FROM regions WHERE name ILIKE :q LIMIT 1"),
            {"q": f"%{region_query}%"},
        )
        row = result.fetchone()

    if row is None:
        return {**state, "error": f"Region '{region_query}' not found in database"}

    region_id: int = row[0]

    async with get_async_session() as session:
        tiles = await session.execute(
            text("""
                SELECT s3_path FROM sentinel2_tiles
                WHERE region_id = :rid
                ORDER BY date DESC
                LIMIT 5
            """),
            {"rid": region_id},
        )
        tile_paths = [r[0] for r in tiles.fetchall()]

    return {
        **state,
        "region_id": region_id,
        "tile_paths": tile_paths,
        "segmentation_results": {},
        "depth_map": {},
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_ingest.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agents/ingest.py tests/agents/test_ingest.py
git commit -m "feat: add Ingest agent node for LangGraph"
```

---

### Task 5: Analysis Agent node

**Files:**
- Create: `src/agents/analysis.py`
- Create: `tests/agents/test_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_analysis.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.analysis import analysis_node


@pytest.mark.asyncio
async def test_analysis_node_sets_risk_tier_and_score():
    mock_session = AsyncMock()
    # forecast query result
    fc_row = MagicMock()
    fc_row.__getitem__ = lambda self, k: {0: {}, 1: True, 2: False}[k]
    # risk_assessment query result
    ra_row = MagicMock()
    ra_row.__getitem__ = lambda self, k: {0: "high", 1: 0.87, 2: 0.74}[k]

    mock_session.execute = AsyncMock(side_effect=[
        MagicMock(fetchone=MagicMock(return_value=fc_row)),
        MagicMock(fetchone=MagicMock(return_value=ra_row)),
    ])

    with (
        patch("src.agents.analysis.run_forecast_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.run_classification_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_id": 1, "region_query": "Harris County"}
        result = await analysis_node(state)

    assert result["risk_tier"] == "high"
    assert result["risk_score"] == pytest.approx(0.74, abs=0.01)
    assert "forecast" in result


@pytest.mark.asyncio
async def test_analysis_node_handles_missing_db_rows():
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(
        return_value=MagicMock(fetchone=MagicMock(return_value=None))
    )

    with (
        patch("src.agents.analysis.run_forecast_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.run_classification_for_region", new_callable=AsyncMock),
        patch("src.agents.analysis.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {"region_id": 99, "region_query": "Unknown"}
        result = await analysis_node(state)

    assert result["risk_tier"] == "moderate"
    assert result["risk_score"] == pytest.approx(0.5, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_analysis.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.analysis'`

- [ ] **Step 3: Create analysis node**

Create `src/agents/analysis.py`:

```python
from sqlalchemy import text
from src.agents.state import ZimRadarState
from src.pipeline.forecasting import run_forecast_for_region
from src.pipeline.classifier import run_classification_for_region
from src.storage.db import get_async_session


async def analysis_node(state: ZimRadarState) -> ZimRadarState:
    region_id = state.get("region_id", 0)

    await run_forecast_for_region(region_id)
    await run_classification_for_region(region_id)

    async with get_async_session() as session:
        fc_result = await session.execute(
            text("""
                SELECT forecast_30d, flood_risk_flag, fire_risk_flag
                FROM forecasts
                WHERE region_id = :rid
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"rid": region_id},
        )
        fc_row = fc_result.fetchone()

    async with get_async_session() as session:
        ra_result = await session.execute(
            text("""
                SELECT risk_tier, confidence, composite_score
                FROM risk_assessments
                WHERE region_id = :rid
                ORDER BY assessed_at DESC
                LIMIT 1
            """),
            {"rid": region_id},
        )
        ra_row = ra_result.fetchone()

    forecast = {
        "forecast_30d": fc_row[0] if fc_row else {},
        "flood_risk_flag": bool(fc_row[1]) if fc_row else False,
        "fire_risk_flag": bool(fc_row[2]) if fc_row else False,
    }

    return {
        **state,
        "forecast": forecast,
        "risk_tier": ra_row[0] if ra_row else "moderate",
        "risk_score": float(ra_row[2]) if ra_row else 0.5,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_analysis.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agents/analysis.py tests/agents/test_analysis.py
git commit -m "feat: add Analysis agent node (forecast + classify)"
```

---

### Task 6: Report Agent node

**Files:**
- Create: `src/agents/report.py`
- Create: `tests/agents/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_report.py
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.report import report_node


@pytest.mark.asyncio
async def test_report_node_sets_draft_and_citations():
    mock_context = [
        {
            "id": 1,
            "text": "Harris County experienced severe flooding in 2023.",
            "source_type": "fema_declaration",
            "source_id": "FEMA-1234",
            "chunk_index": 0,
            "metadata": {"county_fips": "48201"},
            "similarity": 0.92,
            "rerank_score": 1.5,
        }
    ]

    with (
        patch("src.agents.report.retrieve", new_callable=AsyncMock, return_value=mock_context),
        patch("src.agents.report.complete", new_callable=AsyncMock, return_value="Risk is high [1]."),
    ):
        state = {
            "region_query": "Harris County, TX",
            "region_id": 1,
            "risk_tier": "high",
            "risk_score": 0.78,
            "forecast": {"flood_risk_flag": True, "fire_risk_flag": False},
            "retry_count": 0,
        }
        result = await report_node(state)

    assert result["report_draft"] == "Risk is high [1]."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["index"] == 1
    assert result["retrieved_context"] == mock_context


@pytest.mark.asyncio
async def test_report_node_increments_retry_context(monkeypatch):
    with (
        patch("src.agents.report.retrieve", new_callable=AsyncMock, return_value=[]),
        patch("src.agents.report.complete", new_callable=AsyncMock, return_value="Report text."),
    ):
        state = {
            "region_query": "Test Region",
            "region_id": 1,
            "risk_tier": "low",
            "risk_score": 0.2,
            "forecast": {},
            "retry_count": 1,
        }
        result = await report_node(state)

    assert result["report_draft"] == "Report text."
    assert result["retry_count"] == 1  # unchanged by report node
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_report.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.report'`

- [ ] **Step 3: Create report node**

Create `src/agents/report.py`:

```python
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

    context_text = "\n\n".join(
        f"[{i + 1}] {doc['text']}" for i, doc in enumerate(context_docs)
    )

    prompt = f"""Generate a climate risk assessment report for: {region_query}

Risk Assessment:
- Risk Tier: {risk_tier}
- Composite Score: {risk_score:.2f}
- Flood Risk Flag: {forecast.get('flood_risk_flag', False)}
- Fire Risk Flag: {forecast.get('fire_risk_flag', False)}

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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_report.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agents/report.py tests/agents/test_report.py
git commit -m "feat: add Report agent node (RAG + OpenRouter narrative)"
```

---

### Task 7: Validator Agent node

**Files:**
- Create: `src/agents/validator.py`
- Create: `tests/agents/test_validator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_validator.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.validator import validator_node


@pytest.mark.asyncio
async def test_validator_finalizes_when_score_above_threshold():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    with (
        patch("src.agents.validator.complete", new_callable=AsyncMock, return_value="0.92"),
        patch("src.agents.validator.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {
            "region_id": 1,
            "report_draft": "Climate risk is high [1].",
            "retrieved_context": [{"text": "Flooding is common in this region."}],
            "citations": [{"index": 1, "text": "Flooding is common.", "source_type": "fema", "source_id": "123"}],
            "risk_tier": "high",
            "risk_score": 0.8,
            "retry_count": 0,
        }
        result = await validator_node(state)

    assert result["factuality_score"] == pytest.approx(0.92, abs=0.01)
    assert result["final_report"] == "Climate risk is high [1]."
    assert result["low_confidence"] is False
    assert result["report_id"] is not None


@pytest.mark.asyncio
async def test_validator_routes_back_when_score_below_threshold_and_retries_remain():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    with (
        patch("src.agents.validator.complete", new_callable=AsyncMock, return_value="0.55"),
        patch("src.agents.validator.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {
            "region_id": 1,
            "report_draft": "Vague report with no citations.",
            "retrieved_context": [{"text": "Some text."}],
            "citations": [],
            "risk_tier": "low",
            "risk_score": 0.3,
            "retry_count": 0,
        }
        result = await validator_node(state)

    assert result["factuality_score"] == pytest.approx(0.55, abs=0.01)
    assert result["final_report"] is None  # signal to retry
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_validator_finalizes_after_max_retries_with_low_confidence():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()

    with (
        patch("src.agents.validator.complete", new_callable=AsyncMock, return_value="0.55"),
        patch("src.agents.validator.get_async_session") as mock_ctx,
    ):
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        state = {
            "region_id": 1,
            "report_draft": "Still vague.",
            "retrieved_context": [],
            "citations": [],
            "risk_tier": "low",
            "risk_score": 0.3,
            "retry_count": 2,
        }
        result = await validator_node(state)

    assert result["final_report"] == "Still vague."
    assert result["low_confidence"] is True
    assert result["report_id"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_validator.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.validator'`

- [ ] **Step 3: Create validator node**

Create `src/agents/validator.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_validator.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/agents/validator.py tests/agents/test_validator.py
git commit -m "feat: add Validator agent node (LLM-as-judge + DB save)"
```

---

### Task 8: LangGraph graph assembly

**Files:**
- Create: `src/agents/graph.py`
- Create: `tests/agents/test_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_graph.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/agents/test_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.agents.graph'`

- [ ] **Step 3: Create graph module**

Create `src/agents/graph.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/agents/test_graph.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run all agent tests together**

```bash
pytest tests/agents/ -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/agents/graph.py tests/agents/test_graph.py
git commit -m "feat: add LangGraph StateGraph with ingest→analysis→report→validator pipeline"
```

---

### Task 9: FastAPI service

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/main.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_main.py
import uuid
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


def _make_app():
    from src.api.main import app
    return app


def test_assess_returns_report_id():
    report_id = str(uuid.uuid4())
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={
        "final_report": "Risk is high.",
        "report_id": report_id,
        "low_confidence": False,
    })

    with patch("src.api.main.build_graph", return_value=mock_graph):
        client = TestClient(_make_app())
        resp = client.post("/assess", json={"region": "Harris County, TX"})

    assert resp.status_code == 200
    body = resp.json()
    assert "report_id" in body
    assert body["report_id"] == report_id


def test_assess_requires_auth_when_api_key_set(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    from src.config import get_settings
    get_settings.cache_clear()

    with patch("src.api.main.build_graph", return_value=MagicMock()):
        client = TestClient(_make_app())
        resp = client.post("/assess", json={"region": "Test"})

    assert resp.status_code == 401
    get_settings.cache_clear()


def test_get_report_not_found():
    with patch("src.api.main.get_async_session") as mock_ctx:
        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=None)
        mock_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.return_value.__aexit__ = AsyncMock(return_value=None)

        client = TestClient(_make_app())
        resp = client.get(f"/report/{uuid.uuid4()}")

    assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/api/test_main.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.api'`

- [ ] **Step 3: Create package markers and FastAPI app**

Create `src/api/__init__.py` (empty file).

Create `tests/api/__init__.py` (empty file).

Create `src/api/main.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/api/test_main.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ -v --ignore=tests/evals -m "not slow"
```

Expected: all pass (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/api/__init__.py src/api/main.py tests/api/__init__.py tests/api/test_main.py
git commit -m "feat: add FastAPI service with POST /assess and GET /report/{id}"
```

---

### Task 10: Dockerfile and docker-compose update

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add `api` build target to Dockerfile**

Current `Dockerfile` ends with the `streamlit` target. Add an `api` target after it:

```dockerfile
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Full updated `Dockerfile`:

```dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgdal-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system -e .
COPY src/ ./src/

FROM base AS worker
RUN uv pip install --system -e ".[dev]"
COPY tests/ ./tests/
CMD ["python", "-m", "prefect", "worker", "start", "--pool", "default-agent-pool"]

FROM base AS streamlit
EXPOSE 8501
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Add `api` service to docker-compose.yml**

Add after the `streamlit` service block, before the `volumes:` section:

```yaml
  api:
    build:
      context: .
      target: api
    env_file: .env
    environment:
      DATABASE_URL: postgresql+asyncpg://zimradar:${POSTGRES_PASSWORD:-password}@postgres:5432/zimradar
      REDIS_URL: redis://redis:6379
      OLLAMA_URL: http://ollama:11434
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s
```

- [ ] **Step 3: Verify docker-compose config is valid**

```bash
docker compose config --quiet
```

Expected: exits 0 with no errors

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: add api Docker target and docker-compose service on port 8000"
```

---

### Task 11: Streamlit dashboard — wire Run Assessment button

**Files:**
- Modify: `src/dashboard/app.py`

- [ ] **Step 1: Read the current app.py**

```bash
cat src/dashboard/app.py
```

Note the current structure before modifying.

- [ ] **Step 2: Replace the placeholder assess logic with graph invocation**

Find the section that handles the "Run Assessment" button (look for `st.button` or equivalent). Replace the body of the button handler so it calls `build_graph().ainvoke()` and displays results.

The updated button handler should look like this (adapt to wherever the button logic currently lives in the file):

```python
if st.button("Run Assessment", type="primary"):
    if not region_name:
        st.error("Enter a region name first.")
    else:
        import asyncio
        from src.agents.graph import build_graph

        with st.status("Running assessment…", expanded=True) as status:
            st.write("Ingesting region data…")
            initial_state = {
                "region_query": region_name,
                "retry_count": 0,
                "final_report": None,
                "low_confidence": False,
            }
            graph = build_graph()
            try:
                final_state = asyncio.run(graph.ainvoke(initial_state))
                status.update(label="Assessment complete!", state="complete")
            except Exception as exc:
                status.update(label="Assessment failed", state="error")
                st.error(str(exc))
                st.stop()

        if final_state.get("final_report"):
            st.subheader("Risk Assessment")
            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Tier", final_state.get("risk_tier", "—").upper())
            col2.metric("Score", f"{final_state.get('risk_score', 0):.2f}")
            col3.metric(
                "Factuality",
                f"{final_state.get('factuality_score', 0):.2f}",
                delta="⚠ low confidence" if final_state.get("low_confidence") else None,
            )

            st.subheader("Narrative Report")
            st.markdown(final_state["final_report"])

            if final_state.get("citations"):
                st.subheader("Sources")
                for c in final_state["citations"]:
                    st.caption(f"[{c['index']}] {c['source_type']} — {c['source_id']}: {c['text']}")
```

- [ ] **Step 3: Verify the dashboard starts without import errors**

```bash
python -c "import src.dashboard.app"
```

Expected: no import errors (module imports cleanly; Streamlit itself doesn't run in a plain Python import)

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/app.py
git commit -m "feat: wire Streamlit Run Assessment button to LangGraph pipeline"
```

---

## Self-Review Against Spec

### Spec coverage check

| Spec requirement | Task |
|---|---|
| OpenRouter free models as LLM provider | Tasks 1 + 2 |
| No API keys committed to git | Task 1 (env var only, `.env` in `.gitignore`) |
| LangGraph state machine: Ingest → Analysis → Report → Validator | Tasks 3–8 |
| Conditional retry edge (factuality < 0.8, max 2 retries) | Task 7 (validator) + Task 8 (conditional edge) |
| `low_confidence = true` flag when retries exhausted | Task 7 |
| FastAPI `POST /assess` | Task 9 |
| FastAPI `GET /report/{id}` | Task 9 |
| Bearer auth middleware | Task 9 |
| `api` Docker service | Task 10 |
| Streamlit "Run Assessment" calls graph directly | Task 11 |
| Report saved to `reports` table (existing ORM) | Task 7 |

### Placeholder scan

No "TBD", "TODO", or "implement later" phrases found in task steps. All code blocks contain complete implementations.

### Type consistency check

- `ZimRadarState` fields `region_id`, `report_id`, `citations` introduced in Task 3 are used consistently in Tasks 4–9.
- `validate_node` saves `Report` with `id=uuid.UUID(report_id)` — matches ORM `id: Mapped[UUID]`.
- `get_graph()` in `src/api/main.py` references `build_graph()` from `src.agents.graph` — matches Task 8 export.
- `report_node` returns `report_draft` key; `validator_node` reads `state.get("report_draft")` — consistent.
- `run_forecast_for_region` and `run_classification_for_region` signatures match existing `src/pipeline/forecasting.py` and `src/pipeline/classifier.py`.
