import json
import logging
from typing import Any
import redis as redis_lib
from src.config import get_settings

logger = logging.getLogger(__name__)
DEFAULT_TTL = 7 * 24 * 3600  # 7 days

_client: redis_lib.Redis | None = None


def _get_client() -> redis_lib.Redis:
    global _client
    if _client is None:
        _client = redis_lib.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def make_cache_key(s3_path: str, model_version: str) -> str:
    return f"inference:{model_version}:{s3_path}"


def get_cached(key: str) -> dict[str, Any] | None:
    try:
        raw = _get_client().get(key)
    except redis_lib.RedisError:
        logger.warning("Redis GET failed for key %r", key, exc_info=True)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt cache entry for key %r — evicting", key)
        try:
            _get_client().delete(key)
        except redis_lib.RedisError:
            pass
        return None


def set_cached(key: str, value: dict[str, Any], ttl: int = DEFAULT_TTL) -> None:
    try:
        _get_client().setex(key, ttl, json.dumps(value))
    except redis_lib.RedisError:
        logger.warning("Redis SET failed for key %r", key, exc_info=True)
