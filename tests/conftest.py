import os
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from src.storage.models import Base


TEST_DB_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://zimradar:password@localhost:5432/zimradar_test",
)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def db_session(test_engine) -> AsyncSession:
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()
