import fakeredis
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


def test_redis_error_in_get_returns_none():
    from src.storage import cache
    with patch("src.storage.cache._get_client") as mock_get:
        mock_get.return_value.get.side_effect = __import__("redis").RedisError("down")
        assert cache.get_cached("key") is None


def test_redis_error_in_set_is_swallowed():
    from src.storage import cache
    with patch("src.storage.cache._get_client") as mock_get:
        mock_get.return_value.setex.side_effect = __import__("redis").RedisError("down")
        cache.set_cached("key", {"x": 1})  # must not raise


def test_corrupt_cache_entry_returns_none_and_evicts():
    import fakeredis
    from src.storage import cache

    fake = fakeredis.FakeRedis(decode_responses=True)
    fake.set("bad_key", "not-valid-json")
    with patch("src.storage.cache._client", fake):
        result = cache.get_cached("bad_key")
    assert result is None
    assert fake.exists("bad_key") == 0  # key was evicted
