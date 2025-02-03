"""FastAPI server implementation for PyRAI."""

from typing import Optional
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from pyrai.core.config import config
from pyrai.core.state import app_state
from pyrai.core.logging import logger
from pyrai.core.errors import PyRaiError
from pyrai.api.middleware import (
    RequestContextMiddleware,
    RequestMetricsMiddleware,
    RequestLoggingMiddleware
)
from pyrai.api.routes import router
from pyrai.api.config import settings
from pyrai.api.exceptions import APIError
from pyrai.api.responses import create_error
from pyrai.api.middleware import (
    RequestIDMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
)
from .auth import auth_handler
from .middleware import rate_limiter, request_logger
from .tasks import task_queue
from .model_manager import model_manager
from .monitoring import metrics_collector

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    try:
        # Startup
        await app_state.initialize()
        logger.info("Application startup complete")
        yield
    finally:
        # Shutdown
        await app_state.shutdown()
        logger.info("Application shutdown complete")

def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="PyRai API",
        description="AI Model Management API",
        version="0.1.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(RequestMetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # Add error handlers
    @app.exception_handler(PyRaiError)
    async def pyrai_error_handler(request: Request, exc: PyRaiError) -> JSONResponse:
        """Handle PyRai errors."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error(str(exc.detail)),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=create_error(
                "Validation Error",
                detail=str(exc),
                code="VALIDATION_ERROR",
            ),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error(str(exc.detail)),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=create_error(
                "Internal Server Error",
                detail=str(exc) if settings.DEBUG else None,
                code="INTERNAL_ERROR",
            ),
        )

    # Include routers
    app.include_router(router)

    return app

def run_server(
    host: Optional[str] = None,
    port: Optional[int] = None,
    workers: Optional[int] = None,
    reload: bool = False
) -> None:
    """Run API server."""
    uvicorn.run(
        "pyrai.api.server:create_app",
        host=host or config.host,
        port=port or config.port,
        workers=workers or config.workers,
        factory=True,
        reload=reload,
        log_config=None,  # Use our custom logging
    )

app = create_app()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    # Start task processor
    asyncio.create_task(task_queue.process_tasks())
    
    # Load existing models
    model_manager.load_models()
    
    # Start metrics collection
    metrics_collector.start_monitoring()

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to PyRAI API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    run_server() 