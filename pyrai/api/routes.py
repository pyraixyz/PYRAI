"""API routes for PyRAI."""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, Optional
from pydantic import BaseModel

from ..optimization.distillation import (
    KnowledgeDistillation, 
    DistillationConfig,
    DistillationMethod
)

router = APIRouter()

class DistillationRequest(BaseModel):
    """Distillation request schema."""
    teacher_model: Dict[str, Any]
    student_model: Dict[str, Any]
    config: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None

@router.post("/distill")
async def create_distillation(request: DistillationRequest):
    """Create a new distillation task."""
    try:
        config = DistillationConfig(**request.config)
        distiller = KnowledgeDistillation(
            teacher_model=request.teacher_model,
            student_model=request.student_model,
            config=config
        )
        return {"status": "success", "message": "Distillation task created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/methods")
async def list_methods():
    """List available distillation methods."""
    return {
        "methods": [method.value for method in DistillationMethod]
    }

@router.get("/status/{task_id}")
async def get_status(task_id: str):
    """Get status of a distillation task."""
    # TODO: Implement task status tracking
    return {"status": "pending", "task_id": task_id} 