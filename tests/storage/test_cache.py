import fakeredis
import pytest
from unittest.mock import patch


def test_cache_roundtrip():
    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch("src.storage.cache._client", fake):
        from src.storage import cache
        cache.set_cached("key1", {"result": 42})
        assert cache.get_cached("key1") == {"result": 42}


def test_cache_miss_returns_none():
    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch("src.storage.cache._client", fake):
        from src.storage import cache
        assert cache.get_cached("nonexistent") is None


def test_make_cache_key_format():
    from src.storage.cache import make_cache_key
    assert make_cache_key("s3://b/tile.tif", "model-v1") == "inference:model-v1:s3://b/tile.tif"


def test_ttl_default_is_7_days():
    from src.storage.cache import DEFAULT_TTL
    assert DEFAULT_TTL == 7 * 24 * 3600
