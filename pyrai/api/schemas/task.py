from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from pyrai.core.constants import TaskStatus

class TaskBase(BaseModel):
    """Base task schema."""
    name: str
    task_type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    model_id: Optional[int] = None

class TaskCreate(TaskBase):
    """Task creation schema."""
    pass

class TaskUpdate(BaseModel):
    """Task update schema."""
    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    status: Optional[TaskStatus] = None

class TaskResponse(TaskBase):
    """Task response schema."""
    id: int
    user_id: int
    status: TaskStatus
    progress: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""
        from_attributes = True 