from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Awaitable
from datetime import datetime
import asyncio
import json

from pyrai.core.logging import logger
from pyrai.core.cache import cache_manager

T = TypeVar('T')

class EventEmitter:
    """Event emitter for handling system events."""

    def __init__(self) -> None:
        """Initialize event emitter."""
        self.listeners: Dict[str, List[Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]]] = {}
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history = 1000

    def on(
        self,
        event: str,
        callback: Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]
    ) -> None:
        """Register an event listener."""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Remove an event listener."""
        if event in self.listeners:
            self.listeners[event].remove(callback)

    async def emit(self, event: str, data: Any = None) -> None:
        """Emit an event."""
        timestamp = datetime.now(datetime.UTC)
        event_data = {
            "event": event,
            "data": data,
            "timestamp": timestamp.isoformat()
        }

        # Store in history
        if event not in self.history:
            self.history[event] = []
        self.history[event].append(event_data)
        if len(self.history[event]) > self.max_history:
            self.history[event].pop(0)

        # Store in cache
        cache_key = f"event:{event}:{timestamp.timestamp()}"
        await cache_manager.set(cache_key, json.dumps(event_data), expire=86400)

        # Notify listeners
        if event in self.listeners:
            for callback in self.listeners[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")

    def get_history(self, event: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history."""
        if event not in self.history:
            return []
        return self.history[event][-limit:]

    async def replay_events(
        self,
        event: str,
        callback: Callable,
        start_time: Optional[datetime] = None
    ) -> None:
        """Replay historical events."""
        events = self.get_history(event)
        if start_time:
            events = [
                e for e in events
                if datetime.fromisoformat(e["timestamp"]) >= start_time
            ]

        for event_data in events:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_data)
                else:
                    callback(event_data)
            except Exception as e:
                logger.error(f"Error replaying event: {e}")

event_emitter = EventEmitter() 