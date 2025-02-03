from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from pyrai.models.user import User
from pyrai.api.auth import get_current_active_superuser
from pyrai.api.schemas.monitoring import (
    SystemMetrics,
    ModelMetrics,
    TaskMetrics,
    HealthCheck
)
from pyrai.core.metrics import (
    cpu_usage_percent,
    memory_usage_bytes,
    model_inference_total,
    model_training_duration_seconds,
    db_connections_active
)
from pyrai.core.state import app_state

router = APIRouter()

@router.get("/metrics", response_class=Response)
async def get_metrics(
    current_user: User = Depends(get_current_active_superuser)
) -> Any:
    """Get Prometheus metrics."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/system", response_model=SystemMetrics)
async def get_system_metrics(
    current_user: User = Depends(get_current_active_superuser)
) -> Any:
    """Get system metrics."""
    return {
        "cpu_usage": cpu_usage_percent._value.get(),
        "memory_usage": memory_usage_bytes._value.get(),
        "db_connections": db_connections_active._value.get(),
        "uptime_seconds": app_state.get_uptime(),
        "active_tasks": len(app_state.tasks)
    }

@router.get("/models/{model_id}", response_model=ModelMetrics)
async def get_model_metrics(
    model_id: int,
    current_user: User = Depends(get_current_active_superuser)
) -> Any:
    """Get model metrics."""
    model = await app_state.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )

    return {
        "inference_count": model_inference_total.labels(
            model_name=model.name,
            version=model.current_version
        )._value.get(),
        "training_duration": model_training_duration_seconds.labels(
            model_name=model.name,
            version=model.current_version
        )._value.get(),
        "versions": len(model.versions),
        "last_inference": model.last_inference,
        "last_training": model.last_training
    }

@router.get("/tasks/{task_id}", response_model=TaskMetrics)
async def get_task_metrics(
    task_id: int,
    current_user: User = Depends(get_current_active_superuser)
) -> Any:
    """Get task metrics."""
    task = await app_state.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    return {
        "status": task.status,
        "progress": task.progress,
        "duration": task.get_duration(),
        "memory_usage": task.memory_usage,
        "cpu_usage": task.cpu_usage
    }

@router.get("/health", response_model=HealthCheck)
async def health_check() -> Any:
    """Health check endpoint."""
    # Check database connection
    try:
        async with app_state.session() as session:
            await session.execute("SELECT 1")
        db_healthy = True
    except Exception:
        db_healthy = False

    # Check Redis connection
    try:
        async with app_state.redis() as redis:
            await redis.ping()
        cache_healthy = True
    except Exception:
        cache_healthy = False

    # Check task queue
    queue_healthy = await app_state.check_task_queue()

    return {
        "status": "healthy" if all([db_healthy, cache_healthy, queue_healthy]) else "unhealthy",
        "database": db_healthy,
        "cache": cache_healthy,
        "task_queue": queue_healthy,
        "uptime": app_state.get_uptime()
    } 