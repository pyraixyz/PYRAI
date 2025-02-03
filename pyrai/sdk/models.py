"""SDK data models and schemas."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime

class DistillationMethod(str, Enum):
    """Available distillation methods."""
    SOFT_TARGETS = "soft_targets"
    FEATURE_MATCHING = "feature_matching"
    ATTENTION = "attention"
    RELATION = "relation"

class ModelConfig(BaseModel):
    """Model configuration."""
    name: str
    architecture: str
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class DistillationConfig(BaseModel):
    """Distillation configuration."""
    method: DistillationMethod
    temperature: float = Field(default=4.0, gt=0)
    alpha: float = Field(default=0.5, ge=0, le=1)
    feature_layers: Optional[List[str]] = None
    attention_layers: Optional[List[str]] = None

class TaskStatus(BaseModel):
    """Task status information."""
    id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def is_complete(self) -> bool:
        """Check if task is completed."""
        return self.status in ["completed", "failed"]

class DistillationResult(BaseModel):
    """Distillation task results."""
    task_id: str
    teacher_model: str
    student_model: str
    metrics: Dict[str, float]
    training_time: float
    created_at: datetime 