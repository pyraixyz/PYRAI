"""
Training Coordinator implementation for PYRAI.
Handles distributed training coordination and management.
"""

import os
import asyncio
import uuid
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .interfaces import TrainingInterface
from .exceptions import TrainingError, TrainingInterruptedError
from .model_manager import ModelManager
from .resource_scheduler import ResourceScheduler
from .config import config


class TrainingStatus(Enum):
    """Training job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class TrainingJob:
    """Training job details"""
    id: str
    model_id: str
    config: Dict[str, Any]
    resource_id: Optional[str]
    status: TrainingStatus
    progress: float
    start_time: datetime
    end_time: Optional[datetime] = None
    error: Optional[str] = None


class TrainingCoordinator(TrainingInterface):
    """
    Training Coordinator implementation that manages distributed training.
    Handles job scheduling, resource allocation, and progress tracking.
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        resource_scheduler: Optional[ResourceScheduler] = None
    ):
        """
        Initialize the training coordinator.
        
        Args:
            model_manager: Optional model manager instance
            resource_scheduler: Optional resource scheduler instance
        """
        self.model_manager = model_manager or ModelManager()
        self.resource_scheduler = resource_scheduler or ResourceScheduler()
        self.jobs: Dict[str, TrainingJob] = {}
        self._active_jobs: Set[str] = set()
        self._job_queues: Dict[str, asyncio.Queue] = {}

    async def _allocate_resources(self, config: Dict[str, Any]) -> str:
        """Allocate resources for training"""
        try:
            requirements = {
                "cpu_cores": config.get("cpu_cores", 1),
                "memory_mb": config.get("memory_mb", config.min_memory_required),
                "gpu_devices": config.get("gpu_devices", 0)
            }
            return await self.resource_scheduler.allocate(requirements)
        except Exception as e:
            raise TrainingError(f"Failed to allocate resources: {str(e)}")

    async def _setup_training(self, job: TrainingJob) -> None:
        """Setup training environment"""
        try:
            # Load model
            model = await self.model_manager.load(job.model_id)
            
            # Create job queue
            self._job_queues[job.id] = asyncio.Queue()
            
            # Update job status
            job.status = TrainingStatus.RUNNING
            job.start_time = datetime.utcnow()
            
            # Start training task
            asyncio.create_task(self._run_training(job, model))
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            raise TrainingError(f"Failed to setup training: {str(e)}")

    async def _run_training(self, job: TrainingJob, model: Any) -> None:
        """Run training task"""
        try:
            queue = self._job_queues[job.id]
            
            while True:
                # Check for stop signal
                try:
                    message = queue.get_nowait()
                    if message == "stop":
                        job.status = TrainingStatus.INTERRUPTED
                        break
                except asyncio.QueueEmpty:
                    pass
                
                # Update progress (simulated for now)
                job.progress = min(1.0, job.progress + 0.01)
                
                if job.progress >= 1.0:
                    job.status = TrainingStatus.COMPLETED
                    break
                
                await asyncio.sleep(1)  # Simulate training step
                
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            
        finally:
            # Cleanup
            job.end_time = datetime.utcnow()
            if job.resource_id:
                await self.resource_scheduler.deallocate(job.resource_id)
            self._active_jobs.remove(job.id)
            del self._job_queues[job.id]

    async def start_training(self, config: Dict[str, Any]) -> str:
        """
        Start a training job.
        
        Args:
            config: Training configuration including:
                - model_id: Model identifier
                - cpu_cores: Number of CPU cores needed
                - memory_mb: Amount of memory needed
                - gpu_devices: Number of GPU devices needed
                - batch_size: Training batch size
                - learning_rate: Learning rate
                - max_epochs: Maximum epochs
                
        Returns:
            str: Job ID
            
        Raises:
            TrainingError: If starting training fails
        """
        try:
            # Validate config
            if "model_id" not in config:
                raise TrainingError("model_id is required in config")
            
            # Create job
            job_id = str(uuid.uuid4())
            job = TrainingJob(
                id=job_id,
                model_id=config["model_id"],
                config=config,
                resource_id=None,
                status=TrainingStatus.PENDING,
                progress=0.0,
                start_time=datetime.utcnow()
            )
            
            # Allocate resources
            job.resource_id = await self._allocate_resources(config)
            
            # Store job
            self.jobs[job_id] = job
            self._active_jobs.add(job_id)
            
            # Setup and start training
            await self._setup_training(job)
            
            return job_id
            
        except Exception as e:
            raise TrainingError(f"Failed to start training: {str(e)}")

    async def stop_training(self, job_id: str) -> bool:
        """
        Stop a training job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            bool: True if stop was successful
            
        Raises:
            TrainingError: If stopping fails
        """
        try:
            if job_id not in self.jobs:
                return False
                
            job = self.jobs[job_id]
            
            if job.status not in [TrainingStatus.PENDING, TrainingStatus.RUNNING]:
                return False
                
            # Send stop signal
            if job_id in self._job_queues:
                await self._job_queues[job_id].put("stop")
                
            return True
            
        except Exception as e:
            raise TrainingError(f"Failed to stop training: {str(e)}")

    async def get_progress(self, job_id: str) -> Dict[str, Any]:
        """
        Get training progress.
        
        Args:
            job_id: The job identifier
            
        Returns:
            Dictionary containing job progress:
                - status: Current job status
                - progress: Progress percentage
                - start_time: Job start time
                - end_time: Job end time (if completed)
                - error: Error message (if failed)
                
        Raises:
            TrainingError: If progress retrieval fails
        """
        try:
            if job_id not in self.jobs:
                raise TrainingError(f"Job {job_id} not found")
                
            job = self.jobs[job_id]
            return {
                "id": job.id,
                "model_id": job.model_id,
                "status": job.status.value,
                "progress": job.progress,
                "start_time": job.start_time.isoformat(),
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "error": job.error
            }
            
        except Exception as e:
            raise TrainingError(f"Failed to get progress: {str(e)}")

    async def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all training jobs.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of job information dictionaries
            
        Raises:
            TrainingError: If listing fails
        """
        try:
            jobs = []
            for job in self.jobs.values():
                if status and job.status.value != status:
                    continue
                    
                jobs.append({
                    "id": job.id,
                    "model_id": job.model_id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "start_time": job.start_time.isoformat(),
                    "end_time": job.end_time.isoformat() if job.end_time else None
                })
            return jobs
            
        except Exception as e:
            raise TrainingError(f"Failed to list jobs: {str(e)}")

    async def get_job_logs(self, job_id: str) -> List[str]:
        """
        Get logs for a specific job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            List of log messages
            
        Raises:
            TrainingError: If log retrieval fails
        """
        try:
            if job_id not in self.jobs:
                raise TrainingError(f"Job {job_id} not found")
                
            # TODO: Implement actual log collection
            return [
                f"Job {job_id} logs will be implemented in future versions"
            ]
            
        except Exception as e:
            raise TrainingError(f"Failed to get job logs: {str(e)}")

    async def cleanup_jobs(self, max_age_days: int = 7) -> int:
        """
        Clean up old completed jobs.
        
        Args:
            max_age_days: Maximum age of jobs to keep
            
        Returns:
            Number of jobs cleaned up
            
        Raises:
            TrainingError: If cleanup fails
        """
        try:
            now = datetime.utcnow()
            cleanup_count = 0
            
            for job_id in list(self.jobs.keys()):
                job = self.jobs[job_id]
                
                if job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
                    if job.end_time:
                        age = (now - job.end_time).days
                        if age > max_age_days:
                            del self.jobs[job_id]
                            cleanup_count += 1
                            
            return cleanup_count
            
        except Exception as e:
            raise TrainingError(f"Failed to cleanup jobs: {str(e)}") 