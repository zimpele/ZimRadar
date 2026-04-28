import json
import redis as redis_lib
from src.config import get_settings

DEFAULT_TTL = 7 * 24 * 3600  # 7 days

_client: redis_lib.Redis | None = None


def _get_client() -> redis_lib.Redis:
    global _client
    if _client is None:
        _client = redis_lib.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def make_cache_key(s3_path: str, model_version: str) -> str:
    return f"inference:{model_version}:{s3_path}"


def get_cached(key: str) -> dict | None:
    raw = _get_client().get(key)
    return json.loads(raw) if raw else None


def set_cached(key: str, value: dict, ttl: int = DEFAULT_TTL) -> None:
    _get_client().setex(key, ttl, json.dumps(value))
