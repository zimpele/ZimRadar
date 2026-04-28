# tests/rag/test_retriever.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_candidate(text: str, similarity: float = 0.5) -> dict:
    return {
        "id": 1,
        "text": text,
        "source_type": "fema",
        "source_id": "DR-1234",
        "chunk_index": 0,
        "metadata": {"county_fips": "06037"},
        "similarity": similarity,
    }


def test_vec_to_pg_formats_correctly():
    from src.rag.retriever import _vec_to_pg
    result = _vec_to_pg([0.1, -0.2, 0.3])
    assert result.startswith("[")
    assert result.endswith("]")
    parts = result[1:-1].split(",")
    assert len(parts) == 3
    assert abs(float(parts[0]) - 0.1) < 1e-6


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_no_candidates():
    from src.rag.retriever import retrieve

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.rag.retriever.get_async_session", return_value=mock_session),
        patch("src.rag.retriever.TextEmbedder") as mock_emb_cls,
    ):
        mock_emb_cls.return_value.embed.return_value = [0.1] * 384
        result = await retrieve("flood risk in Los Angeles")

    assert result == []


@pytest.mark.asyncio
async def test_retrieve_reranks_and_returns_top_k():
    from src.rag.retriever import retrieve

    rows = [
        MagicMock(id=i, chunk_text=f"text {i}", source_type="fema",
                  source_id="DR-100", chunk_index=i, metadata={}, similarity=0.5)
        for i in range(10)
    ]

    mock_session = MagicMock()
    mock_session.execute = AsyncMock(return_value=iter(rows))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Cross-encoder gives highest score to item 7
    ce_scores = [float(i) for i in range(10)]
    ce_scores[7] = 99.0

    with (
        patch("src.rag.retriever.get_async_session", return_value=mock_session),
        patch("src.rag.retriever.TextEmbedder") as mock_emb_cls,
        patch("src.rag.retriever._get_cross_encoder") as mock_ce_factory,
    ):
        mock_emb_cls.return_value.embed.return_value = [0.1] * 384
        mock_ce_factory.return_value.predict.return_value = ce_scores
        result = await retrieve("query", top_k=3)

    assert len(result) == 3
    assert result[0]["text"] == "text 7"  # highest rerank score
    assert "rerank_score" in result[0]


@pytest.mark.asyncio
async def test_retrieve_builds_metadata_filter_clause():
    from src.rag.retriever import retrieve

    executed_sqls = []
    mock_session = MagicMock()
    mock_session.execute = AsyncMock(
        side_effect=lambda sql, params=None: (
            executed_sqls.append(str(sql)) or iter([])
        )
    )
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("src.rag.retriever.get_async_session", return_value=mock_session),
        patch("src.rag.retriever.TextEmbedder") as mock_emb_cls,
    ):
        mock_emb_cls.return_value.embed.return_value = [0.1] * 384
        await retrieve("query", county_fips="06037", disaster_type="flood")

    assert len(executed_sqls) == 1
    assert "county_fips" in executed_sqls[0]
    assert "disaster_type" in executed_sqls[0]
