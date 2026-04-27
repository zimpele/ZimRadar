from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from src.config import get_settings


def _make_engine():
    settings = get_settings()
    return create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)


_engine = None
_session_factory = None


def _ensure_initialized():
    global _engine, _session_factory
    if _engine is None:
        _engine = _make_engine()
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


@asynccontextmanager
async def get_async_session() -> AsyncSession:
    _ensure_initialized()
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
