from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from pyrai.core.constants import TaskStatus

class SystemMetrics(BaseModel):
    """System metrics schema."""
    cpu_usage: float
    memory_usage: int
    db_connections: int
    uptime_seconds: int
    active_tasks: int

class ModelMetrics(BaseModel):
    """Model metrics schema."""
    inference_count: int
    training_duration: float
    versions: int
    last_inference: Optional[datetime] = None
    last_training: Optional[datetime] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

class TaskMetrics(BaseModel):
    """Task metrics schema."""
    status: TaskStatus
    progress: Optional[int] = None
    duration: float
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

class HealthCheck(BaseModel):
    """Health check schema."""
    status: str
    database: bool
    cache: bool
    task_queue: bool
    uptime: int
    details: Optional[Dict[str, Any]] = None 