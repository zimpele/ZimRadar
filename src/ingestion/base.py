import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Awaitable, TypeVar
from sqlalchemy import text
from src.storage.db import get_async_session

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 1.0,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {exc}. Retrying in {delay}s")
                await asyncio.sleep(delay)
    raise last_exc


async def log_failure(flow_name: str, error_message: str, region_id: int | None = None) -> None:
    async with get_async_session() as session:
        await session.execute(
            text(
                "INSERT INTO failed_ingestion (region_id, flow_name, error_message, failed_at) "
                "VALUES (:region_id, :flow_name, :error_message, :failed_at)"
            ),
            {
                "region_id": region_id,
                "flow_name": flow_name,
                "error_message": error_message,
                "failed_at": datetime.now(timezone.utc),
            },
        )
