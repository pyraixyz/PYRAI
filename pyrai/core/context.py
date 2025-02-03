from typing import Any, Dict, Optional
from contextvars import ContextVar
from uuid import UUID, uuid4
from datetime import datetime, timezone

request_id: ContextVar[UUID] = ContextVar('request_id', default=uuid4())
user_id: ContextVar[Optional[int]] = ContextVar('user_id', default=None)
trace_id: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)
start_time: ContextVar[datetime] = ContextVar('start_time', default=datetime.now(timezone.utc))
extra_context: ContextVar[Dict[str, Any]] = ContextVar('extra_context', default={})

def get_request_id() -> UUID:
    """Get current request ID."""
    return request_id.get()

def set_request_id(value: UUID) -> None:
    """Set current request ID."""
    request_id.set(value)

def get_user_id() -> Optional[int]:
    """Get current user ID."""
    return user_id.get()

def set_user_id(value: Optional[int]) -> None:
    """Set current user ID."""
    user_id.set(value)

def get_trace_id() -> Optional[str]:
    """Get current trace ID."""
    return trace_id.get()

def set_trace_id(value: Optional[str]) -> None:
    """Set current trace ID."""
    trace_id.set(value)

def get_start_time() -> datetime:
    """Get request start time."""
    return start_time.get()

def set_start_time(value: datetime) -> None:
    """Set request start time."""
    start_time.set(value)

def get_context() -> Dict[str, Any]:
    """Get full request context."""
    return {
        'request_id': str(get_request_id()),
        'user_id': get_user_id(),
        'trace_id': get_trace_id(),
        'start_time': get_start_time().isoformat(),
        **extra_context.get()
    }

def set_context_value(key: str, value: Any) -> None:
    """Set a context value."""
    context = extra_context.get().copy()
    context[key] = value
    extra_context.set(context)

def get_context_value(key: str, default: Any = None) -> Any:
    """Get a context value."""
    return extra_context.get().get(key, default)

def clear_context() -> None:
    """Clear all context values."""
    request_id.set(uuid4())
    user_id.set(None)
    trace_id.set(None)
    start_time.set(datetime.now(timezone.utc))
    extra_context.set({}) 