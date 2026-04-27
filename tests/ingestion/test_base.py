import pytest
from src.ingestion.base import with_retry, log_failure  # noqa: F401


@pytest.mark.asyncio
async def test_with_retry_succeeds_on_first_try():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await with_retry(flaky, max_attempts=3)
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_with_retry_retries_on_failure():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("transient error")
        return "ok"

    result = await with_retry(flaky, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_with_retry_raises_after_max_attempts():
    async def always_fails():
        raise ValueError("permanent error")

    with pytest.raises(ValueError, match="permanent error"):
        await with_retry(always_fails, max_attempts=2, base_delay=0.01)
