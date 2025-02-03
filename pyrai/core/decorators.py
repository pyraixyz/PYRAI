from typing import Any, Callable, TypeVar, cast, Optional
from functools import wraps
import time
import asyncio

from pyrai.core.logging import logger
from pyrai.core.metrics import (
    request_duration_seconds,
    cache_hits_total,
    cache_misses_total
)
from pyrai.core.cache import cache_manager

F = TypeVar('F', bound=Callable[..., Any])

def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable[[F], F]:
    """Retry decorator with exponential backoff."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
            raise last_exception  # type: ignore

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
            raise last_exception  # type: ignore

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)
    return decorator

def cache(
    ttl: int = 300,
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None
) -> Callable[[F], F]:
    """Cache decorator."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"

            cached_value = await cache_manager.get(cache_key)
            if cached_value is not None:
                cache_hits_total.labels(cache_name=func.__name__).inc()
                return cached_value

            cache_misses_total.labels(cache_name=func.__name__).inc()
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, expire=ttl)
            return result

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args) + str(kwargs))}"

            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                cache_hits_total.labels(cache_name=func.__name__).inc()
                return cached_value

            cache_misses_total.labels(cache_name=func.__name__).inc()
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, expire=ttl)
            return result

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)
    return decorator

def timing(name: str) -> Callable[[F], F]:
    """Timing decorator."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                request_duration_seconds.labels(
                    method=name,
                    path=func.__name__
                ).observe(duration)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                request_duration_seconds.labels(
                    method=name,
                    path=func.__name__
                ).observe(duration)

        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper)
    return decorator 