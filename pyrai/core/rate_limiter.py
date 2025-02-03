from typing import Dict, Optional, Tuple, Protocol
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

from pyrai.core.cache import cache_manager
from pyrai.core.logging import logger

class AsyncCache(Protocol):
    async def zremrangebyscore(self, name: str, min: float, max: float) -> int: ...
    async def zcount(self, name: str, min: float, max: str) -> int: ...
    async def zadd(self, name: str, mapping: Dict[str, float]) -> int: ...
    async def expire(self, name: str, time: int) -> bool: ...

@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests: int
    window: int  # seconds
    key_prefix: str = "ratelimit"

class RateLimiter:
    """Rate limiter using sliding window."""

    def __init__(self) -> None:
        """Initialize rate limiter."""
        self.limits: Dict[str, RateLimit] = {}
        self.cache: AsyncCache = cache_manager

    def add_limit(self, name: str, requests: int, window: int) -> None:
        """Add a rate limit."""
        self.limits[name] = RateLimit(requests=requests, window=window)

    async def is_allowed(
        self,
        name: str,
        key: str,
        cost: int = 1
    ) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed."""
        if name not in self.limits:
            return True, None

        limit = self.limits[name]
        cache_key = f"{limit.key_prefix}:{name}:{key}"

        try:
            # Get current window data
            now = datetime.now(datetime.UTC)
            window_start = int(now.timestamp()) - limit.window

            # Clean old data
            await self.cache.zremrangebyscore(cache_key, 0, window_start)

            # Get current count
            count = await self.cache.zcount(cache_key, window_start, "+inf")

            if count + cost > limit.requests:
                retry_after = limit.window - (int(now.timestamp()) - window_start)
                return False, retry_after

            # Add new request
            await self.cache.zadd(cache_key, {str(now.timestamp()): cost})
            await self.cache.expire(cache_key, limit.window * 2)

            return True, None

        except Exception as e:
            logger.error(f"Rate limit error: {e}")
            return True, None

    async def get_remaining(self, name: str, key: str) -> Tuple[int, int]:
        """Get remaining requests and reset time."""
        if name not in self.limits:
            return -1, 0

        limit = self.limits[name]
        cache_key = f"{limit.key_prefix}:{name}:{key}"

        try:
            now = datetime.now(datetime.UTC)
            window_start = int(now.timestamp()) - limit.window

            # Clean old data
            await self.cache.zremrangebyscore(cache_key, 0, window_start)

            # Get current count
            count = await self.cache.zcount(cache_key, window_start, "+inf")
            remaining = max(0, limit.requests - count)
            reset = limit.window - (int(now.timestamp()) - window_start)

            return remaining, reset

        except Exception as e:
            logger.error(f"Rate limit error: {e}")
            return -1, 0

rate_limiter = RateLimiter() 