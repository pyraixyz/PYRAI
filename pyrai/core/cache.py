from typing import Any, Optional
import json
from datetime import datetime, timedelta

from redis import Redis
from redis.exceptions import RedisError

from pyrai.api.config import settings
from pyrai.core.logging import logger

class CacheManager:
    """Redis cache manager."""

    def __init__(self) -> None:
        """Initialize cache manager."""
        self._redis: Optional[Redis] = None

    @property
    def redis(self) -> Redis:
        """Get Redis client."""
        if self._redis is None:
            self._redis = Redis.from_url(
                settings.REDIS_URL,
                db=settings.REDIS_DB,
                decode_responses=True,
            )
        return self._redis

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            data = self.redis.get(f"{settings.REDIS_PREFIX}{key}")
            return json.loads(data) if data else None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        """Set value in cache."""
        try:
            full_key = f"{settings.REDIS_PREFIX}{key}"
            data = json.dumps(value)
            if nx:
                return bool(self.redis.setnx(full_key, data))
            self.redis.set(full_key, data, ex=expire)
            return True
        except (RedisError, TypeError) as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            return bool(self.redis.delete(f"{settings.REDIS_PREFIX}{key}"))
        except RedisError as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(self.redis.exists(f"{settings.REDIS_PREFIX}{key}"))
        except RedisError as e:
            logger.error(f"Cache exists error: {e}")
            return False

    def ttl(self, key: str) -> int:
        """Get TTL of key in cache."""
        try:
            return self.redis.ttl(f"{settings.REDIS_PREFIX}{key}")
        except RedisError as e:
            logger.error(f"Cache TTL error: {e}")
            return -1

    def incr(self, key: str) -> Optional[int]:
        """Increment value in cache."""
        try:
            return self.redis.incr(f"{settings.REDIS_PREFIX}{key}")
        except RedisError as e:
            logger.error(f"Cache increment error: {e}")
            return None

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiry on key."""
        try:
            return bool(self.redis.expire(f"{settings.REDIS_PREFIX}{key}", seconds))
        except RedisError as e:
            logger.error(f"Cache expire error: {e}")
            return False

cache_manager = CacheManager()

def get_redis_client() -> Redis:
    """Get Redis client."""
    return cache_manager.redis 