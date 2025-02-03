from typing import Any, Dict, Optional, TypeVar
from datetime import datetime, timezone
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from pyrai.core.config import config
from pyrai.db.session import async_session
from pyrai.core.cache import cache_manager
from pyrai.core.logging import logger

T = TypeVar("T")

class AppState:
    """Application state manager."""

    def __init__(self) -> None:
        """Initialize application state."""
        self.initialized: bool = False
        self.start_time: datetime = datetime.now(timezone.utc)
        self._state: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def initialize(self) -> None:
        """Initialize application state."""
        if self.initialized:
            return

        try:
            # Initialize database
            async with async_session() as session:
                await session.execute("SELECT 1")

            # Initialize cache
            await cache_manager.ping()

            self.initialized = True
            logger.info("Application state initialized")
        except Exception as e:
            logger.error(f"Failed to initialize application state: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown application state."""
        if not self.initialized:
            return

        try:
            # Cleanup code here
            self.initialized = False
            logger.info("Application state shutdown complete")
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set state value."""
        self._state[key] = value

    def delete(self, key: str) -> None:
        """Delete state value."""
        self._state.pop(key, None)

    @asynccontextmanager
    async def session(self) -> AsyncSession:
        """Get database session."""
        async with async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @asynccontextmanager
    async def redis(self) -> Redis:
        """Get Redis connection."""
        redis = await cache_manager.get_connection()
        try:
            yield redis
        finally:
            await redis.close()

    async def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a lock."""
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

app_state = AppState() 