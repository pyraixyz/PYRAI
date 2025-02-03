from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from pyrai.core.constants import ModelType, ModelFramework, ModelStatus

class ModelBase(BaseModel):
    """Base model schema."""
    name: str
    description: Optional[str] = None
    model_type: ModelType
    framework: ModelFramework
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ModelCreate(ModelBase):
    """Model creation schema."""
    pass

class ModelUpdate(BaseModel):
    """Model update schema."""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[ModelStatus] = None

class ModelVersion(BaseModel):
    """Model version schema."""
    id: int
    model_id: int
    version: str
    file_path: str
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""
        from_attributes = True

class ModelResponse(ModelBase):
    """Model response schema."""
    id: int
    user_id: int
    status: ModelStatus
    current_version: Optional[str] = None
    versions: list[ModelVersion] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""
        from_attributes = True 