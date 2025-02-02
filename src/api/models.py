"""
API data models for PYRAI.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class TrainingConfig(BaseModel):
    """Training configuration model"""
    model_id: str = Field(..., description="Model identifier")
    cpu_cores: int = Field(1, description="Number of CPU cores needed")
    memory_mb: int = Field(1024, description="Amount of memory needed in MB")
    gpu_devices: int = Field(0, description="Number of GPU devices needed")
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(0.001, description="Learning rate")
    max_epochs: int = Field(100, description="Maximum number of epochs")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "model-123",
                "cpu_cores": 2,
                "memory_mb": 4096,
                "gpu_devices": 1,
                "batch_size": 64,
                "learning_rate": 0.001,
                "max_epochs": 100
            }
        }


class JobResponse(BaseModel):
    """Job response model"""
    id: str = Field(..., description="Job identifier")
    model_id: str = Field(..., description="Model identifier")
    status: str = Field(..., description="Current job status")
    progress: float = Field(..., description="Training progress percentage")
    start_time: datetime = Field(..., description="Job start time")
    end_time: Optional[datetime] = Field(None, description="Job end time")
    error: Optional[str] = Field(None, description="Error message if failed")


class JobList(BaseModel):
    """Job list response model"""
    jobs: List[JobResponse] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details") 