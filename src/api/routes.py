"""
API routes for PYRAI.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from .models import TrainingConfig, JobResponse, JobList, ErrorResponse
from ..core.training_coordinator import TrainingCoordinator
from ..core.exceptions import TrainingError


# Create router
router = APIRouter(prefix="/api/v1")

# Global training coordinator instance
_training_coordinator: Optional[TrainingCoordinator] = None


async def get_coordinator() -> TrainingCoordinator:
    """Get or create training coordinator instance"""
    global _training_coordinator
    if _training_coordinator is None:
        _training_coordinator = TrainingCoordinator()
    return _training_coordinator


@router.post("/training/jobs", response_model=JobResponse, responses={400: {"model": ErrorResponse}})
async def start_training(
    config: TrainingConfig,
    coordinator: TrainingCoordinator = Depends(get_coordinator)
):
    """
    Start a new training job.
    
    Args:
        config: Training configuration
        coordinator: Training coordinator instance
        
    Returns:
        Job information
        
    Raises:
        HTTPException: If starting training fails
    """
    try:
        job_id = await coordinator.start_training(config.dict())
        return await coordinator.get_progress(job_id)
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/training/jobs/{job_id}", response_model=dict, responses={404: {"model": ErrorResponse}})
async def stop_training(
    job_id: str,
    coordinator: TrainingCoordinator = Depends(get_coordinator)
):
    """
    Stop a training job.
    
    Args:
        job_id: Job identifier
        coordinator: Training coordinator instance
        
    Returns:
        Success status
        
    Raises:
        HTTPException: If stopping training fails
    """
    try:
        success = await coordinator.stop_training(job_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return {"success": True}
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/training/jobs/{job_id}", response_model=JobResponse, responses={404: {"model": ErrorResponse}})
async def get_job_progress(
    job_id: str,
    coordinator: TrainingCoordinator = Depends(get_coordinator)
):
    """
    Get training job progress.
    
    Args:
        job_id: Job identifier
        coordinator: Training coordinator instance
        
    Returns:
        Job progress information
        
    Raises:
        HTTPException: If getting progress fails
    """
    try:
        return await coordinator.get_progress(job_id)
    except TrainingError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/training/jobs", response_model=JobList)
async def list_jobs(
    status: Optional[str] = None,
    coordinator: TrainingCoordinator = Depends(get_coordinator)
):
    """
    List training jobs.
    
    Args:
        status: Optional status filter
        coordinator: Training coordinator instance
        
    Returns:
        List of jobs
        
    Raises:
        HTTPException: If listing jobs fails
    """
    try:
        jobs = await coordinator.list_jobs(status)
        return {
            "jobs": jobs,
            "total": len(jobs)
        }
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/training/jobs/{job_id}/logs", response_model=List[str], responses={404: {"model": ErrorResponse}})
async def get_job_logs(
    job_id: str,
    coordinator: TrainingCoordinator = Depends(get_coordinator)
):
    """
    Get training job logs.
    
    Args:
        job_id: Job identifier
        coordinator: Training coordinator instance
        
    Returns:
        List of log messages
        
    Raises:
        HTTPException: If getting logs fails
    """
    try:
        return await coordinator.get_job_logs(job_id)
    except TrainingError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/training/cleanup", response_model=dict)
async def cleanup_jobs(
    max_age_days: int = 7,
    coordinator: TrainingCoordinator = Depends(get_coordinator)
):
    """
    Clean up old training jobs.
    
    Args:
        max_age_days: Maximum age of jobs to keep
        coordinator: Training coordinator instance
        
    Returns:
        Number of jobs cleaned up
        
    Raises:
        HTTPException: If cleanup fails
    """
    try:
        cleaned = await coordinator.cleanup_jobs(max_age_days)
        return {
            "cleaned": cleaned,
            "success": True
        }
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e)) 