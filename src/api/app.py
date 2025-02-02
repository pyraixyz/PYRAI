"""
FastAPI application for PYRAI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router
from ..core.exceptions import TrainingError


# Create FastAPI app
app = FastAPI(
    title="PYRAI API",
    description="API for managing distributed AI training jobs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
app.include_router(router)


@app.exception_handler(TrainingError)
async def training_error_handler(request, exc):
    """Handle training errors"""
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PYRAI API",
        "version": "1.0.0",
        "status": "running"
    } 