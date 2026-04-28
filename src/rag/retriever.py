# src/rag/retriever.py
import asyncio
import threading
from typing import Any
from sentence_transformers import CrossEncoder
from sqlalchemy import text
from src.storage.db import get_async_session
from src.rag.embed import TextEmbedder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_INITIAL = 20
TOP_K_FINAL = 5

_cross_encoder: CrossEncoder | None = None
_cross_encoder_lock = threading.Lock()


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


_text_embedder: TextEmbedder | None = None
_embedder_lock = threading.Lock()


def _get_text_embedder() -> TextEmbedder:
    global _text_embedder
    if _text_embedder is None:
        with _embedder_lock:
            if _text_embedder is None:
                _text_embedder = TextEmbedder()
    return _text_embedder


def _vec_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


async def retrieve(
    query: str,
    *,
    county_fips: str | None = None,
    disaster_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = TOP_K_FINAL,
) -> list[dict[str, Any]]:
    embedder = _get_text_embedder()
    query_vec = await asyncio.to_thread(embedder.embed, query)
    vec_str = _vec_to_pg(query_vec)

    filter_clauses: list[str] = []
    params: dict[str, Any] = {"embedding": vec_str, "limit": TOP_K_INITIAL}

    if county_fips:
        filter_clauses.append("(metadata->>'county_fips') = :county_fips")
        params["county_fips"] = county_fips
    if disaster_type:
        filter_clauses.append("(metadata->>'disaster_type') = :disaster_type")
        params["disaster_type"] = disaster_type
    if date_from:
        filter_clauses.append("(metadata->>'date')::date >= :date_from::date")
        params["date_from"] = date_from
    if date_to:
        filter_clauses.append("(metadata->>'date')::date <= :date_to::date")
        params["date_to"] = date_to

    where = "WHERE " + " AND ".join(filter_clauses) if filter_clauses else ""

    sql = text(f"""
        SELECT id, chunk_text, source_type, source_id, chunk_index, metadata,
               1 - (embedding <=> :embedding::vector) AS similarity
        FROM text_embeddings
        {where}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :limit
    """)

    async with get_async_session() as session:
        result = await session.execute(sql, params)
        candidates = [
            {
                "id": row.id,
                "text": row.chunk_text,
                "source_type": row.source_type,
                "source_id": row.source_id,
                "chunk_index": row.chunk_index,
                "metadata": row.metadata or {},
                "similarity": float(row.similarity),
            }
            for row in result
        ]

    if not candidates:
        return []

    cross_encoder = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = await asyncio.to_thread(cross_encoder.predict, pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
