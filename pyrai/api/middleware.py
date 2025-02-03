"""Middleware for rate limiting and request logging."""

from fastapi import Request, HTTPException
from datetime import datetime, timedelta
from collections import defaultdict
import time
import logging
import structlog
from typing import Dict, Deque, Awaitable, Callable
from collections import deque
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from pyrai.core.context import (
    set_request_id,
    set_user_id,
    set_trace_id,
    set_start_time,
    clear_context,
    get_context
)
from pyrai.core.logging import logger
from pyrai.core.metrics import (
    request_total,
    request_duration_seconds
)

# Configure structured logging
logger = structlog.get_logger()

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=requests_per_minute))
        
    async def check_rate_limit(self, request: Request):
        """Check if request is within rate limit."""
        client_ip = request.client.host
        now = time.time()
        
        # Clean old requests
        while self.requests[client_ip] and \
              self.requests[client_ip][0] < now - 60:
            self.requests[client_ip].popleft()
            
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
            
        self.requests[client_ip].append(now)
        return True

class RequestLogger:
    """Request logging middleware."""
    
    async def log_request(self, request: Request, call_next):
        """Log request and response details."""
        start_time = time.time()
        
        # Log request
        logger.info(
            "request_started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        duration = time.time() - start_time
        logger.info(
            "request_completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration
        )
        
        return response

rate_limiter = RateLimiter()
request_logger = RequestLogger()

class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware for managing request context."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        # Clear previous context
        clear_context()

        # Set request context
        set_request_id(uuid4())
        set_trace_id(request.headers.get("X-Trace-ID"))
        
        try:
            response = await call_next(request)
            return response
        finally:
            clear_context()

class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.time() - start_time
            request_total.labels(
                method=request.method,
                path=request.url.path,
                status=status
            ).inc()
            request_duration_seconds.labels(
                method=request.method,
                path=request.url.path
            ).observe(duration)
        
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint
    ) -> Response:
        context = get_context()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"context": context}
        )
        
        try:
            response = await call_next(request)
            logger.info(
                f"Request completed: {response.status_code}",
                extra={"context": context}
            )
            return response
        except Exception as e:
            logger.error(
                f"Request failed: {str(e)}",
                extra={"context": context},
                exc_info=True
            )
            raise 