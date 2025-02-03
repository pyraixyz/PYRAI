"""API schemas and data models."""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class ModelConfig(BaseModel):
    """Model configuration schema."""
    name: str
    parameters: Dict[str, Any]
    architecture: Optional[str] = None

class DistillationConfigSchema(BaseModel):
    """Distillation configuration schema."""
    temperature: float = Field(default=4.0, gt=0)
    alpha: float = Field(default=0.5, ge=0, le=1)
    method: str
    feature_layers: Optional[List[str]] = None
    attention_layers: Optional[List[str]] = None

class DistillationResponse(BaseModel):
    """Distillation response schema."""
    task_id: str
    status: str
    message: Optional[str] = None 